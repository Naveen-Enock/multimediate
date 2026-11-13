"""Concordance Correlation Coefficient (exact), with masking.

CCC = 2 * cov(x, y) / (var_x + var_y + (mu_x - mu_y)**2)

Equivalent to the spec form 2*rho*sx*sy / (sx^2 + sy^2 + (mu_x-mu_y)^2) since
rho*sx*sy = cov(x, y). Population (biased) variance/covariance are used.

Masked or non-finite positions are excluded from every reduction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ccc"]


def ccc(y_true, y_pred, valid_mask=None, eps: float = 1e-8) -> float:
    """Exact CCC between two 1-D (flattened) sequences.

    Args:
        y_true, y_pred: array-likes of identical shape.
        valid_mask: optional bool/0-1 array, same shape; False/0 positions are
            dropped. NaN/inf positions in either input are also dropped.
        eps: denominator floor to avoid 0/0 on constant inputs.

    Returns:
        CCC as a Python float, or ``nan`` if fewer than 2 valid positions remain.
    """
    x = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    finite = np.isfinite(x) & np.isfinite(y)
    if valid_mask is not None:
        m = np.asarray(valid_mask).reshape(-1).astype(bool)
        if m.shape != finite.shape:
            raise ValueError(f"mask shape {m.shape} != data shape {finite.shape}")
        finite &= m

    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")

    mx, my = x.mean(), y.mean()
    vx = x.var()  # population variance (ddof=0)
    vy = y.var()
    cov = ((x - mx) * (y - my)).mean()

    denom = vx + vy + (mx - my) ** 2
    if denom < eps:
        return float("nan")
    return float(2.0 * cov / denom)
