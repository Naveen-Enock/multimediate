"""Losses for the continuous regression head: masked MSE + CCC (``RegressionCCCLoss``)."""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean squared error over valid horizon positions only.

    pred/target/valid_mask: (B, K). Returns a scalar tensor.
    """
    se = (pred - target) ** 2 * valid_mask
    denom = valid_mask.sum().clamp_min(eps)
    return se.sum() / denom


def differentiable_ccc(pred: torch.Tensor, target: torch.Tensor,
                       valid_mask: torch.Tensor | None = None,
                       eps: float = 1e-8,
                       weight: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable Concordance Correlation Coefficient over the valid positions.

    ``CCC = 2·σ_xy / (σ_x² + σ_y² + (μ_x − μ_y)²)``. Mirrors the exact ``metrics.ccc``
    (population/biased variance) but in torch so it carries gradient. Computed over all valid
    (batch × horizon) elements pooled together; returns a 0-D tensor. Falls back to a (grad-carrying)
    zero when fewer than 2 valid positions remain so a degenerate batch never produces NaN.

    ``weight`` (broadcastable to ``pred``) turns this into a **weighted** CCC — the moments
    ``μ/σ²/σ_xy`` become weight-normalized. Invalid positions are still dropped.
    """
    x = pred.reshape(-1)
    y = target.reshape(-1)
    if weight is not None:
        w = weight.reshape(-1).clamp_min(0.0)
        if valid_mask is not None:
            w = w * (valid_mask.reshape(-1) > 0.5)
        W = w.sum()
        if (w > 0).sum() < 2 or W <= eps:
            return x.sum() * 0.0
        mx = (w * x).sum() / W
        my = (w * y).sum() / W
        vx = (w * (x - mx) ** 2).sum() / W
        vy = (w * (y - my) ** 2).sum() / W
        cov = (w * (x - mx) * (y - my)).sum() / W
        return 2.0 * cov / (vx + vy + (mx - my) ** 2 + eps)
    if valid_mask is not None:
        m = valid_mask.reshape(-1) > 0.5
        x, y = x[m], y[m]
    if x.numel() < 2:
        return x.sum() * 0.0
    mx, my = x.mean(), y.mean()
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    return 2.0 * cov / (vx + vy + (mx - my) ** 2 + eps)


def grouped_mse(pred: torch.Tensor, target: torch.Tensor, group_id: torch.Tensor,
                weight_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean of the per-group masked MSE, segregated by ``group_id`` (mirrors :func:`grouped_ccc`).

    ``weight_mask`` is the same soft (validity × horizon-decay) weight ``masked_mse`` takes. A batch
    that mixes datasets with different label scales/means (e.g. NoXi vs. NoXi-J) no longer lets one
    dataset's squared error dominate the pooled gradient — each dataset's MSE is computed
    independently, then the groups are averaged evenly (not frame-count-weighted, matching
    ``grouped_ccc``). Falls back to a (grad-carrying) zero when no group has any weighted position.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    g = group_id.reshape(-1)
    w = weight_mask.reshape(-1)
    mses = []
    for gid in torch.unique(g):
        wm = w * (g == gid)
        denom = wm.sum()
        if denom <= eps:
            continue
        mses.append(((pred - target) ** 2 * wm).sum() / denom)
    if not mses:
        return pred.sum() * 0.0
    return torch.stack(mses).mean()


def grouped_ccc(pred: torch.Tensor, target: torch.Tensor, group_id: torch.Tensor,
                valid_mask: torch.Tensor | None = None, eps: float = 1e-8,
                weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of the per-group differentiable CCC, segregated by ``group_id``.

    Computes the differentiable CCC within each ``group_id`` (per-element dataset id) and averages
    over the groups that retain ≥2 valid positions. Falls back to a (grad-carrying) zero when no
    group qualifies. ``pred``/``target``/``group_id``/``valid_mask`` are flattened together, so any
    broadcastable shape is accepted.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    g = group_id.reshape(-1)
    vm = None if valid_mask is None else (valid_mask.reshape(-1) > 0.5)
    w = None if weight is None else weight.reshape(-1)
    cccs = []
    for gid in torch.unique(g):
        m = g == gid
        if vm is not None:
            m = m & vm
        if m.sum() < 2:
            continue
        wm = None if w is None else w[m]
        cccs.append(differentiable_ccc(pred[m], target[m], weight=wm, eps=eps))
    if not cccs:
        return pred.sum() * 0.0
    return torch.stack(cccs).mean()


class RegressionCCCLoss(nn.Module):
    """Continuous regression loss: ``L = μ·L_MSE(ŷ, y₀) + λ·(1 − CCC(ŷ, y₀))``.

    ``μ`` (``mse_weight``) defaults to 1.0; set toward 0 to approach the pure-CCC objective (DAPA,
    MM'25 winner, trains ``L = 1 − CCC`` with no MSE term at all). Both terms are dataset-segregated
    via ``group_id`` (``grouped_mse`` / ``grouped_ccc``, each averaged evenly over the datasets
    present in the batch) so a batch mixing datasets with different label scales/means (e.g. NoXi vs.
    NoXi-J) can't have one dataset's error dominate the other's gradient; ``horizon_weight`` decay on
    both terms. ``ŷ`` is the head's raw-scale per-frame prediction; ``y_0`` the raw target chunk.
    """

    def __init__(self, lam: float = 1.0, mse_weight: float = 1.0):
        super().__init__()
        self.lam = float(lam)
        self.mse_weight = float(mse_weight)

    def forward(self, pred: torch.Tensor, y_0: torch.Tensor,
                valid_mask: torch.Tensor | None = None,
                group_id: torch.Tensor | None = None,
                horizon_weight: torch.Tensor | None = None,
                return_parts: bool = False):
        if valid_mask is None:
            valid_mask = torch.ones_like(pred)
        mse_mask = valid_mask if horizon_weight is None else valid_mask * horizon_weight
        ccc_w = None if horizon_weight is None else (valid_mask * horizon_weight)
        if group_id is None:
            mse = masked_mse(pred, y_0, mse_mask)
            ccc = differentiable_ccc(pred, y_0, valid_mask, weight=ccc_w)
        else:
            gid = group_id.expand_as(pred)
            mse = grouped_mse(pred, y_0, gid, mse_mask)
            ccc = grouped_ccc(pred, y_0, gid, valid_mask, weight=ccc_w)
        loss = self.mse_weight * mse + self.lam * (1.0 - ccc)
        if return_parts:
            return loss, {"mse": mse.detach(), "ccc": ccc.detach()}
        return loss
