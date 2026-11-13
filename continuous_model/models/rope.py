"""Rotary position embeddings (RoPE) for the temporal transformers.

Applied to the query/key tensors inside the masked self-attention of the reactive/anticipatory
streams. No learnable parameters; cos/sin tables are cached per (seq_len, device, dtype).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """Rotary embeddings for ``(B, H, T, head_dim)`` q/k tensors. ``head_dim`` must be even."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self._cache: dict = {}

    def _cos_sin(self, T: int, device, dtype):
        key = (T, device, dtype)
        cs = self._cache.get(key)
        if cs is None:
            inv_freq = 1.0 / (self.base ** (
                torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
            t = torch.arange(T, device=device).float()
            freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)         # (T, head_dim)
            cs = (emb.cos().to(dtype), emb.sin().to(dtype))
            self._cache[key] = cs
        return cs

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """q, k: ``(B, H, T, head_dim)`` -> rotated q, k of the same shape."""
        T = q.shape[-2]
        cos, sin = self._cos_sin(T, q.device, q.dtype)
        cos = cos[None, None]                               # (1, 1, T, head_dim)
        sin = sin[None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k
