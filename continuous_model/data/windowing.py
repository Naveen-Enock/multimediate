"""Sliding-window enumeration and target chunking.

Slides a width-W window with stride S across each session and chunks the per-window
target Y_t in R^K.
"""

from __future__ import annotations

import numpy as np
import torch


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

