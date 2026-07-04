"""Unweighted Cohen's kappa (nominal agreement), with masking.

kappa = (p_o - p_e) / (1 - p_e)

where p_o is observed agreement and p_e is chance agreement computed from the
marginal label distributions. No linear/quadratic (QWK) weighting — exact class
match only.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cohen_kappa"]


def cohen_kappa(y_true, y_pred, num_classes: int | None = None,
                valid_mask=None, eps: float = 1e-12) -> float:
    """Unweighted Cohen's kappa between two integer label sequences.

    Args:
        y_true, y_pred: integer array-likes of identical shape.
        num_classes: optional fixed class count; inferred from data if None.
        valid_mask: optional bool/0-1 array; False/0 positions dropped. Negative
            labels (a common ignore-index sentinel) are also dropped.
        eps: floor for the (1 - p_e) denominator.

    Returns:
        kappa as a Python float, or ``nan`` if no valid positions remain.
        Returns 1.0 when chance agreement is perfect and observed agreement
        matches it (degenerate single-class case).
    """
    t = np.asarray(y_true).reshape(-1).astype(np.int64)
    p = np.asarray(y_pred).reshape(-1).astype(np.int64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")

    keep = (t >= 0) & (p >= 0)
    if valid_mask is not None:
        m = np.asarray(valid_mask).reshape(-1).astype(bool)
        if m.shape != keep.shape:
            raise ValueError(f"mask shape {m.shape} != data shape {keep.shape}")
        keep &= m

    t, p = t[keep], p[keep]
    if t.size == 0:
        return float("nan")

    if num_classes is None:
        num_classes = int(max(t.max(), p.max())) + 1
    C = num_classes

    cm = np.zeros((C, C), dtype=np.float64)
    np.add.at(cm, (t, p), 1.0)

    n = cm.sum()
    p_o = np.trace(cm) / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    p_e = float((row * col).sum())

    denom = 1.0 - p_e
    if abs(denom) < eps:
        # No chance-corrected room; perfect-by-chance agreement -> kappa 1.0.
        return 1.0
    return float((p_o - p_e) / denom)
