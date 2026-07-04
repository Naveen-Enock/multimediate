"""Temporal ensembling — merge overlapping chunk predictions per frame.

Overlapping windows (stride ``S`` < horizon ``K``) produce several chunk predictions covering each
frame. Merge them with an exponential decay on the within-chunk horizon offset
``Δt = frame - chunk_start``:

    w(Δt) = exp(-κ · Δt)

so near-horizon predictions are weighted higher. Continuous values are merged directly before CCC;
categorical paths merge raw **logits** before ``argmax`` (then unweighted kappa). ``κ`` is
``ensemble.decay_kappa`` in the config.
"""

from __future__ import annotations

import numpy as np


def decay_weights(K: int, kappa: float) -> np.ndarray:
    """``w(Δt) = exp(-κ·Δt)`` for horizon offsets ``Δt = 0..K-1``."""
    return np.exp(-float(kappa) * np.arange(int(K), dtype=np.float64))


def ensemble_continuous(starts, chunks, length: int, kappa: float, masks=None) -> np.ndarray:
    """Merge overlapping continuous chunks into a per-frame prediction.

    ``starts``: (N,) chunk start frame indices; ``chunks``: (N, K) predictions; ``length``: number
    of frames to fill; ``masks``: optional (N, K) — only positions with mask>0.5 contribute.
    Returns ``(length,)``.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    N, K = chunks.shape
    w = decay_weights(K, kappa)
    m = np.ones_like(chunks) if masks is None else np.asarray(masks, dtype=np.float64)
    num = np.zeros(length, dtype=np.float64)
    den = np.zeros(length, dtype=np.float64)
    for i in range(N):
        t = int(starts[i])
        lo, hi = max(t, 0), min(t + K, length)
        if hi <= lo:
            continue
        off = lo - t
        wm = w[off:hi - t] * m[i, off:hi - t]
        num[lo:hi] += wm * chunks[i, off:hi - t]
        den[lo:hi] += wm
    # 1e-15 (not 1e-8) so frames covered only by a single late-horizon chunk position aren't zeroed.
    return num / np.maximum(den, 1e-15)


def subsample_forward_fill(arr: np.ndarray, stride: int) -> np.ndarray:
    """Forward-fill a per-frame array from its 1 Hz subsample (every ``stride``-th frame).

    Each frame ``t`` takes the value sampled at ``(t // stride) * stride`` — i.e. the head's
    macro-granularity (1 Hz) predictions are nearest-neighbour upsampled (held constant) across
    each ``stride``-frame block back onto the 25 fps timeline. Operates on the leading frame axis,
    so it accepts ``(K,)`` labels or ``(K, C)`` logits. ``stride <= 1`` is a no-op.
    """
    if stride <= 1:
        return arr
    K = arr.shape[0]
    idx = (np.arange(K) // stride) * stride
    return arr[idx]


def ensemble_logits(starts, logits, length: int, kappa: float, masks=None) -> np.ndarray:
    """Merge overlapping categorical logits into per-frame logits ``(length, C)`` (argmax later).

    Overlapping windows are merged with the same horizon decay ``w(Δt) = exp(-κ·Δt)`` as the
    continuous path, masked to valid positions, so near-horizon predictions weigh higher.
    """
    logits = np.asarray(logits, dtype=np.float64)
    N, K, C = logits.shape
    w = decay_weights(K, kappa)
    m = np.ones((N, K)) if masks is None else np.asarray(masks, dtype=np.float64)
    num = np.zeros((length, C), dtype=np.float64)
    den = np.zeros((length, 1), dtype=np.float64)
    for i in range(N):
        t = int(starts[i])
        lo, hi = max(t, 0), min(t + K, length)
        if hi <= lo:
            continue
        off = lo - t
        wm = (w[off:hi - t] * m[i, off:hi - t])[:, None]
        num[lo:hi] += wm * logits[i, off:hi - t]
        den[lo:hi] += wm
    return num / np.maximum(den, 1e-15)


def frame_coverage(starts, length: int, K: int, masks=None) -> np.ndarray:
    """Boolean ``(length,)``: frames covered by at least one valid chunk position."""
    cov = np.zeros(length, dtype=bool)
    for i in range(len(starts)):
        t = int(starts[i])
        lo, hi = max(t, 0), min(t + K, length)
        if hi <= lo:
            continue
        if masks is None:
            cov[lo:hi] = True
        else:
            mk = np.asarray(masks[i], dtype=np.float64)
            cov[lo:hi] |= mk[lo - t:hi - t] > 0.5
    return cov
