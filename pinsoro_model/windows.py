"""Window enumeration on the native 30 fps PInSoRo label grid, plus label aggregation.

Windows are enumerated on the annotation grid (30 fps) and every other stream is sliced by the
matching *time interval* rather than resampled -- audio at 25 fps keeps its own indexing, so no
interpolation is introduced anywhere in this pipeline.
"""

from __future__ import annotations

import numpy as np

from .columns import VIDEO_FPS

W_SEC = 10.0        # primary window: below the shortest median label run (17.8 s, cc task)
HOP_SEC = 2.0       # 5x overlap; the decode averages the overlapping posteriors
CONTEXT_SEC = (30.0, 60.0)   # nested multi-scale windows sharing the same centre

W_FRAMES = int(round(W_SEC * VIDEO_FPS))       # 300
HOP_FRAMES = int(round(HOP_SEC * VIDEO_FPS))   # 60


def window_bounds(n_frames: int, w: int = W_FRAMES, s: int = HOP_FRAMES):
    """``(start, end, t_center)`` arrays for width-``w`` stride-``s`` windows over ``n_frames``.

    Strides from 0 and always appends a final
    start at ``n_frames - w`` so the session tail is covered. Sessions shorter than one window yield
    a single truncated window.
    """
    if n_frames <= w:
        start = np.array([0], dtype=np.int64)
    else:
        start = np.arange(0, n_frames - w + 1, s, dtype=np.int64)
        if start[-1] != n_frames - w:
            start = np.append(start, n_frames - w)
    end = np.minimum(start + w, n_frames)
    t_center = (start + end) / 2.0 / VIDEO_FPS
    return start, end, t_center.astype(np.float32)


def context_bounds(t_center: np.ndarray, span_sec: float, n_frames: int):
    """``(start, end)`` on the 30 fps grid for ``span_sec``-wide windows centred on ``t_center``."""
    half = span_sec * VIDEO_FPS / 2.0
    c = t_center * VIDEO_FPS
    start = np.clip(np.round(c - half), 0, n_frames).astype(np.int64)
    end = np.clip(np.round(c + half), 0, n_frames).astype(np.int64)
    return start, end


def time_slice(start30: int, end30: int, sr: float, n: int):
    """Map a 30 fps frame interval to the index range of a stream sampled at ``sr`` with ``n`` rows.

    Returns ``(lo, hi)`` with ``lo < hi`` clamped into ``[0, n]``; ``(0, 0)`` when the interval
    falls entirely outside the stream.
    """
    lo = int(np.floor(start30 / VIDEO_FPS * sr))
    hi = int(np.ceil(end30 / VIDEO_FPS * sr))
    lo = max(0, min(lo, n))
    hi = max(0, min(hi, n))
    return (lo, hi) if hi > lo else (0, 0)


def label_mode_purity(lab: np.ndarray, start: np.ndarray, end: np.ndarray, n_classes: int):
    """Per-window modal class and purity of an int label array (``-1`` = unannotated).

    Returns ``(mode int8 (n_win,), purity float32 (n_win,))``; a window with no valid label gets
    ``mode = -1`` and ``purity = 0``. Purity (the modal class's share of the valid frames) is used
    as the training sample weight, so boundary-straddling windows are down-weighted rather than
    dropped -- dropping them would bias the model against transitions.
    """
    n_win = len(start)
    mode = np.full(n_win, -1, dtype=np.int8)
    purity = np.zeros(n_win, dtype=np.float32)
    for i in range(n_win):
        seg = lab[start[i]:end[i]]
        seg = seg[seg >= 0]
        if seg.size == 0:
            continue
        counts = np.bincount(seg, minlength=n_classes)
        k = int(counts.argmax())
        mode[i] = k
        purity[i] = counts[k] / seg.size
    return mode, purity


def run_lengths(lab: np.ndarray):
    """``(values, lengths)`` of the constant runs in ``lab``, ignoring ``-1`` stretches.

    Used to estimate the HMM transition matrix at decode time and to report the effective
    independent-sample count.
    """
    valid = lab >= 0
    if not valid.any():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    idx = np.flatnonzero(valid)
    v = lab[idx]
    brk = np.flatnonzero(np.diff(idx) != 1) + 1          # gaps in the valid stretch
    chg = np.flatnonzero(np.diff(v) != 0) + 1            # class changes
    cuts = np.unique(np.concatenate([[0], brk, chg, [len(v)]]))
    lens = np.diff(cuts)
    return v[cuts[:-1]], lens
