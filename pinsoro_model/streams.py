"""SSI ``.stream`` reader (XML header + companion ``.stream~`` float32 binary)."""

from __future__ import annotations

import re

import numpy as np

# Some streams carry sentinel/garbage magnitudes (e.g. PInSoRo eGeMAPSv2 is ~10% filled with
# ~3.69e19). Treat |x| above this as invalid.
FEATURE_CLIP = 1e8


def _header(path: str) -> tuple[int, float, int]:
    with open(path, "rb") as f:
        text = f.read(1024).decode("utf-8", errors="ignore")
    dim = int(re.search(r'dim="(\d+)"', text).group(1))
    sr = float(re.search(r'sr="([\d.]+)"', text).group(1))
    num = int(re.search(r'num="(\d+)"', text).group(1))
    return num, sr, dim


def read_stream_header(path: str) -> tuple[int, float, int]:
    """Parse the header only -> ``(num_frames, sr, dim)``."""
    return _header(path)


def read_stream(path: str):
    """Read a stream -> ``(data[num_frames, dim] float32, sr)``."""
    num, sr, dim = _header(path)
    with open(path + "~", "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw[: num * dim * 4], dtype=np.float32).reshape(num, dim)
    return data.astype(np.float32), sr
