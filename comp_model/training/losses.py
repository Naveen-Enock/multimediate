"""Losses, routed by dataset kind.

masked MSE for the continuous regression baseline; an unweighted cross-entropy for the PInSoRo
categorical head; the hybrid diffusion noise + CCC loss for the conditional diffusion transformer
head (``DiffusionHybridCCCLoss``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean squared error over valid horizon positions only.

    pred/target/valid_mask: (B, K). Returns a scalar tensor.
    """
    se = (pred - target) ** 2 * valid_mask
    denom = valid_mask.sum().clamp_min(eps)
    return se.sum() / denom


def masked_ce(logits: torch.Tensor, target: torch.Tensor,
              log_prior: torch.Tensor | None = None,
              sampler_alpha: float = 0.5) -> torch.Tensor:
    """Cross-entropy over valid horizon positions, with the power-sampler-harmonized margin.

    Every frame is scored on its own (no class re-weighting, no batch-level coupling); reduction is
    the plain mean over valid frames. Class balance in the *batch* is handled upstream by the power
    sampler; the margin computed here handles *decision-boundary* calibration in the loss.

    ``log_prior: (B, C)`` (or None) is the per-domain τ-scaled class log-prior ``τ_d·log π_c^nat``
    (the caller bakes the per-domain strength ``τ_d`` into the row). When supplied, the class logits
    are shifted before the softmax by the margin that is in symbolic harmony with the
    ``ClassBalancedSampler``'s ``f_c^(1−α)`` class reshaping (``α = sampler_alpha``):

        m_c = −α·(τ_d·log π_c^nat)   =   α·τ_d·log(1 / π_c^nat)

    The sampler flattens the class marginal toward ``π^(1−α)``; at ``τ_d=1`` the margin supplies the
    remaining ``α`` via ``−α·log π``, so the raw logits read at inference (``log_prior=None``) sit on
    the full natural prior ``π^1``. ``log_prior=None`` (or a zero/τ_d=0 row) ⇒ plain unweighted CE.

    ``logits: (B, K, C)``; ``target: (B, K)`` int64 with ``-1`` at padded/unknown positions
    (ignored). Returns a scalar tensor; yields a grad-carrying zero if a batch has no valid targets.
    """
    B, K, C = logits.shape
    if log_prior is not None:
        margin = -sampler_alpha * log_prior                  # (B, C): −α·(τ_d·log π_c^nat)
        logits = logits + margin.unsqueeze(1)                # (B, 1, C) broadcast over K
    flat_logits = logits.reshape(B * K, C)
    flat_target = target.reshape(B * K)
    if (flat_target != -1).sum() == 0:
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits, flat_target, ignore_index=-1)


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


class DiffusionHybridCCCLoss(nn.Module):
    """Combined DDPM velocity-prediction MSE + correlation loss for the diffusion transformer head.

    ``L = L_MSE(v_pred, v) + λ · (1 − CCC(ŷ₀, y₀))`` where ``(v_pred, v)`` are the predicted and true
    velocities (``v = √ᾱ_t·ε − √(1−ᾱ_t)·y₀``) and the clean engagement trajectory ŷ₀ is analytically
    recovered from ``v_pred`` for the CCC term:

        ŷ₀ = √ᾱ_t·y_t − √(1−ᾱ_t)·v_pred

    The velocity MSE keeps the score-matching objective intact; the CCC term optimizes the
    correlation metric on the reconstructed clean chunk. Both terms are masked to valid horizon
    positions. ``lam`` (λ) defaults to 1.0.

    The head trains in a per-dataset standardized target space (``y_t``/``v_pred`` standardized), so
    the analytically recovered ŷ₀ lands in that space too; the per-element ``y_mean``/``y_std``
    un-standardize it back to the raw engagement scale before CCC. When omitted they default to
    ``(0, 1)`` (identity).

    The CCC is segregated by dataset via ``group_id`` (per-element dataset id); the velocity MSE
    stays pooled.
    """

    def __init__(self, lam: float = 1.0, y_mean: float = 0.0, y_std: float = 1.0):
        super().__init__()
        self.lam = float(lam)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def forward(self, v_pred: torch.Tensor, v: torch.Tensor, y_t: torch.Tensor,
                y_0: torch.Tensor, alpha_bar_t: torch.Tensor,
                valid_mask: torch.Tensor | None = None,
                y_mean: torch.Tensor | float | None = None,
                y_std: torch.Tensor | float | None = None,
                group_id: torch.Tensor | None = None,
                horizon_weight: torch.Tensor | None = None,
                return_parts: bool = False):
        """``v_pred``/``v``/``y_t``/``y_0``: ``(B, K)`` (or ``(B, N, K)``); ``alpha_bar_t`` trailing
        dim 1; ``valid_mask`` matching ``v_pred`` or None -> scalar loss. ``v_pred``/``v`` are
        velocities; ``y_0`` is the raw (un-standardized) target chunk.

        ``y_mean``/``y_std`` (per-element tensors broadcastable to ``v_pred``, or scalars) un-standardize
        the recovered ŷ₀ to raw scale per dataset; default to the scalar ``(self.y_mean, self.y_std)``.
        ``group_id`` (broadcastable to ``v_pred``) segregates the CCC by dataset; None pools it.

        ``horizon_weight`` (broadcastable to ``v_pred``, e.g. ``(1, 1, K)``) softly weights both the
        velocity MSE and the CCC moments per horizon offset. None ⇒ uniform.

        With ``return_parts=True`` also returns ``{"v_mse", "ccc"}`` (detached) so the
        trainer can log the velocity MSE and the CCC term separately."""
        if valid_mask is None:
            valid_mask = torch.ones_like(v_pred)
        mse_mask = valid_mask if horizon_weight is None else valid_mask * horizon_weight
        mse = masked_mse(v_pred, v, mse_mask)
        acp = alpha_bar_t.clamp_min(1e-8)
        y0_hat = acp.sqrt() * y_t - (1.0 - acp).sqrt() * v_pred   # velocity -> ŷ₀ (standardized)
        ys = self.y_std if y_std is None else y_std
        ym = self.y_mean if y_mean is None else y_mean
        y0_hat = y0_hat * ys + ym                         # standardized ŷ₀ -> raw engagement scale
        ccc_w = None if horizon_weight is None else (valid_mask * horizon_weight)
        if group_id is None:
            ccc = differentiable_ccc(y0_hat, y_0, valid_mask, weight=ccc_w)
        else:
            ccc = grouped_ccc(y0_hat, y_0, group_id.expand_as(y0_hat), valid_mask, weight=ccc_w)
        loss = mse + self.lam * (1.0 - ccc)
        if return_parts:
            return loss, {"v_mse": mse.detach(), "ccc": ccc.detach()}
        return loss


class RegressionCCCLoss(nn.Module):
    """Deterministic-regression continuous loss: ``L = L_MSE(ŷ, y₀) + λ·(1 − CCC(ŷ, y₀))``.

    The ``use_diffusion=false`` counterpart of ``DiffusionHybridCCCLoss`` — same objective shaping
    (dataset-segregated CCC via ``group_id``, ``horizon_weight`` decay on both terms) minus the
    velocity/score-matching machinery. ``ŷ`` is the head's raw-scale per-frame prediction; ``y_0``
    the raw target chunk.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = float(lam)

    def forward(self, pred: torch.Tensor, y_0: torch.Tensor,
                valid_mask: torch.Tensor | None = None,
                group_id: torch.Tensor | None = None,
                horizon_weight: torch.Tensor | None = None,
                return_parts: bool = False):
        if valid_mask is None:
            valid_mask = torch.ones_like(pred)
        mse_mask = valid_mask if horizon_weight is None else valid_mask * horizon_weight
        mse = masked_mse(pred, y_0, mse_mask)
        ccc_w = None if horizon_weight is None else (valid_mask * horizon_weight)
        if group_id is None:
            ccc = differentiable_ccc(pred, y_0, valid_mask, weight=ccc_w)
        else:
            ccc = grouped_ccc(pred, y_0, group_id.expand_as(pred), valid_mask, weight=ccc_w)
        loss = mse + self.lam * (1.0 - ccc)
        if return_parts:
            return loss, {"mse": mse.detach(), "ccc": ccc.detach()}
        return loss
