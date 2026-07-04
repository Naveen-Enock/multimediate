"""Stream + Whisper I/O and 25 fps resampling.

Ports ``read_stream`` and ``resample_col`` from
``correlation/noxi_dim_correlation.py`` (the canonical SSI ``.stream``/``.stream~``
reader and linear resampler) so all loaders share one implementation. Whisper
``.npy`` features are already 25 fps (see ``extract_whisper.py``); they are only
length-aligned here, not re-interpolated.
"""

from __future__ import annotations

import os
import re

import numpy as np
from scipy.interpolate import interp1d

TARGET_SR = 25.0  # TARGET_FPS — every node/modality lands on this grid.

# Some streams carry sentinel/garbage magnitudes (e.g. PInSoRo eGeMAPSv2 is ~10%
# filled with ~3.69e19; NoXi eGeMAPS maxes near 2e6). Treat |x| above this as
# invalid. All real modalities (whisper/swin/clip ~O(1), openface/openpose <1e5,
# egemaps <1e7) sit well below it.
FEATURE_CLIP = 1e8

__all__ = ["read_stream", "read_stream_header", "resample_to_len", "load_stream",
           "load_whisper", "valid_feature_mask", "sanitize_feature", "FEATURE_CLIP",
           "read_str_csv_to_idx"]


def read_stream_header(path: str) -> tuple[int, float, int]:
    """Parse an SSI ``.stream`` header only -> ``(num_frames, sr, dim)``.

    Reads the XML header without touching the (potentially large) ``.stream~`` binary, so it is a
    cheap way to recover a session's native frame count + sample rate (used to derive the 25 fps
    label-grid length for test sessions, which ship no annotation CSVs).
    """
    with open(path, "rb") as f:
        text = f.read(1024).decode("utf-8", errors="ignore")
    dim = int(re.search(r'dim="(\d+)"', text).group(1))
    sr = float(re.search(r'sr="([\d.]+)"', text).group(1))
    num = int(re.search(r'num="(\d+)"', text).group(1))
    return num, sr, dim


def read_str_csv_to_idx(path: str, class_map: dict) -> np.ndarray:
    """Map a one-label-per-line string CSV to int64 class indices (-1 = unknown)."""
    idx = []
    with open(path) as f:
        for line in f:
            s = line.strip().lower()
            if not s:
                continue
            idx.append(class_map.get(s, -1))
    return np.asarray(idx, dtype=np.int64)


def valid_feature_mask(arr, clip: float = FEATURE_CLIP, valid_min: float | None = None):
    """Bool mask of finite, in-range feature values.

    ``valid_min`` (per-modality, optional) rejects values below a floor — used for
    OpenPose, whose undetected keypoints/confidences carry a ``-1`` sentinel while every
    real value (pixel coordinate, confidence) is non-negative.
    """
    m = np.isfinite(arr) & (np.abs(arr) <= clip)
    if valid_min is not None:
        m &= (arr >= valid_min)
    return m


def sanitize_feature(arr, clip: float = FEATURE_CLIP):
    """Zero out non-finite / out-of-range values (masked-neutral)."""
    return np.where(valid_feature_mask(arr, clip), arr, 0.0).astype(np.float32)


def read_stream(path: str):
    """Read an SSI ``.stream`` header + companion ``.stream~`` binary.

    Returns ``(data[nf, dim] float32, sr)``. Ported verbatim from
    ``correlation/noxi_dim_correlation.py`` to keep parsing identical.
    """
    with open(path, "rb") as f:
        text = f.read(1024).decode("utf-8", errors="ignore")
    dim = int(re.search(r'dim="(\d+)"', text).group(1))
    sr = float(re.search(r'sr="([\d.]+)"', text).group(1))
    nf = int(re.search(r'num="(\d+)"', text).group(1))
    with open(path + "~", "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw[: nf * dim * 4], dtype=np.float32).reshape(nf, dim)
    return data.astype(np.float32), sr


def resample_to_len(data, src_sr, tgt_len, session_dur):
    """Resample every column of ``data[nf, dim]`` to ``(tgt_len, dim)``.

    Vectorized along the dim axis (single interp1d over axis=0); semantics match
    the per-column ``resample_col`` in the correlation script, including the
    short-duration source-rate re-inference guard.

    FOR FEATURE STREAMS ONLY. Do NOT pass categorical/integer label arrays here —
    linear interp creates fractional, non-existent class indices at boundaries.
    Use ``windowing.resample_categorical_nearest`` for PInSoRo social/task labels.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected (nf, dim), got {data.shape}")
    src_len, dim = data.shape
    if src_len == tgt_len:
        return data
    if src_len < 2:
        row = data[0] if src_len else np.zeros(dim, np.float32)
        return np.tile(row, (tgt_len, 1)).astype(np.float32)
    declared_dur = src_len / src_sr
    if declared_dur < session_dur * 0.95:
        src_sr = src_len / session_dur
    src_t = np.arange(src_len) / src_sr
    tgt_t = np.arange(tgt_len) / TARGET_SR
    f = interp1d(src_t, data, kind="linear", axis=0, bounds_error=False,
                 fill_value=(data[0], data[-1]))
    return f(tgt_t).astype(np.float32)


def load_stream(path: str, tgt_len: int, session_dur: float):
    """Load a ``.stream`` and resample it to ``(tgt_len, dim)`` at 25 fps."""
    data, sr = read_stream(path)
    return resample_to_len(data, sr, tgt_len, session_dur)


def load_whisper(path: str, tgt_len: int):
    """Load a Whisper ``.npy`` (already 25 fps) and length-align to ``tgt_len``.

    Truncates if longer; edge-pads (repeat last frame) if shorter, matching the
    pad/truncate convention in ``extract_whisper.py``.
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T = arr.shape[0]
    if T == tgt_len:
        return arr
    if T > tgt_len:
        return arr[:tgt_len]
    pad = np.repeat(arr[-1:], tgt_len - T, axis=0)
    return np.concatenate([arr, pad], axis=0)


def file_exists(path: str) -> bool:
    """A stream is present iff both header and ``~`` binary exist; npy is a file."""
    if path.endswith(".stream"):
        return os.path.exists(path) and os.path.exists(path + "~")
    return os.path.exists(path)
