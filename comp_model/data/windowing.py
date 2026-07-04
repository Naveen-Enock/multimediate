"""Sliding-window enumeration, target chunking, and categorical resampling.

Slides a width-W window with stride S across each session and chunks the per-window
target Y_t in R^K.

``resample_categorical_nearest`` must be used for all integer-class label arrays
(PInSoRo social / task). NEVER pass categorical labels to ``streams.resample_to_len``
— that function uses linear interp and will produce fractional, non-existent class
indices at class-transition boundaries (e.g. 2.5 between class 2 and class 3).
"""

from __future__ import annotations

import numpy as np
import torch


def resample_categorical_nearest(arr: np.ndarray, tgt_len: int) -> np.ndarray:
    """Nearest-index resample of a 1-D integer label array to ``tgt_len``.

    Uses floor-based index mapping so the output is always a valid member of the
    original class set — safe for PInSoRo social/task integer labels at any frame-rate
    ratio (e.g. 30 fps → 25 fps). Do NOT use ``streams.resample_to_len`` for categoricals;
    it applies linear interpolation and produces fractional class indices at boundaries.
    """
    L = len(arr)
    if L == tgt_len or L == 0:
        return arr
    src_idx = np.floor(np.arange(tgt_len) * (L / tgt_len)).astype(np.int64)
    src_idx = np.clip(src_idx, 0, L - 1)
    return arr[src_idx]


def window_starts(T: int, W: int, S: int) -> list[int]:
    """Start indices of width-W, stride-S windows covering ``[0, T)``.

    Sessions no longer than a window yield a single zero-padded window. Otherwise
    we stride from 0 and always append a final start at ``T - W`` so the tail is
    covered even when ``(T - W)`` is not a multiple of ``S``.
    """
    if T <= W:
        return [0]
    starts = list(range(0, T - W + 1, S))
    if starts[-1] != T - W:
        starts.append(T - W)
    return starts


def frames_valid(T: int, start: int, W: int) -> torch.Tensor:
    """(W,) float mask: 1 for real frames, 0 where the window runs past ``T``."""
    fv = np.zeros(W, dtype=np.float32)
    fv[: max(0, min(W, T - start))] = 1.0
    return torch.from_numpy(fv)


def chunk_continuous(lab: np.ndarray, start: int, K: int):
    """Y_t = ``lab[start:start+K]`` padded to K; mask is 1 on finite frames."""
    chunk = np.full(K, np.nan, dtype=np.float32)
    n = max(0, min(K, len(lab) - start))
    if n > 0:
        chunk[:n] = lab[start:start + n]
    mask = np.isfinite(chunk).astype(np.float32)
    chunk = np.nan_to_num(chunk, nan=0.0)
    return torch.from_numpy(chunk), torch.from_numpy(mask)


def chunk_categorical(lab: np.ndarray, start: int, K: int):
    """Class-index chunk padded with -1; mask is 1 where the label is >= 0."""
    chunk = np.full(K, -1, dtype=np.int64)
    n = max(0, min(K, len(lab) - start))
    if n > 0:
        chunk[:n] = lab[start:start + n]
    mask = (chunk >= 0).astype(np.float32)
    return torch.from_numpy(chunk), torch.from_numpy(mask)
