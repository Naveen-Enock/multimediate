"""Reactive + anticipatory temporal transformers.

Two isolated single-pass masked transformer streams over the time axis T, each
``num_encoder_layers`` AdaLN-Zero blocks with RoPE inside the masked self-attention:

  * Reactive (forward-causal, lower-triangular): each frame attends to itself + earlier frames.
  * Anticipatory (backward-causal, upper-triangular): each frame attends to itself + later frames.

Each block uses standard Pre-LN with a zero-initialized per-channel gate on each residual
(``adaln.AdaLNZero``), so each sub-layer starts as identity. Both streams consume the gated-pool
sequence ``(B, T, d)`` and return ``(B, T, d)``. Padded frames are masked as keys; the diagonal is
always kept so no query row is fully masked (-> NaN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaln import AdaLNZero
from .rope import RoPE


def _temporal_mask(direction: str, frames_valid: torch.Tensor) -> torch.Tensor:
    """``(B, 1, T, T)`` boolean attend-mask (True = attend): causal direction + key padding.

    ``forward``/``backward`` are the causal reactive/anticipatory directions; ``bidirectional`` is
    the non-causal single-stream mode (every frame attends to every valid frame).
    """
    B, T = frames_valid.shape
    idx = torch.arange(T, device=frames_valid.device)
    if direction == "forward":
        causal = idx[None, :] <= idx[:, None]          # allow j <= i (lower triangular)
    elif direction == "backward":
        causal = idx[None, :] >= idx[:, None]          # allow j >= i (upper triangular)
    else:  # bidirectional — no temporal direction, only key padding constrains attention
        causal = torch.ones(T, T, dtype=torch.bool, device=frames_valid.device)
    key_valid = frames_valid > 0.5                     # (B, T)
    allowed = causal[None] & key_valid[:, None, :]     # (B, T, T)
    eye = torch.eye(T, dtype=torch.bool, device=frames_valid.device)[None]
    allowed = allowed | eye                            # keep diagonal -> no all-masked row
    return allowed[:, None]                            # (B, 1, T, T), broadcasts over heads


class MaskedMHSA(nn.Module):
    """Multi-head self-attention with RoPE on Q/K and an additive boolean attend-mask."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, rope: RoPE | None):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = rope

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, h, T, dh)
        if self.rope is not None:
            q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class AdaLNZeroBlock(nn.Module):
    """AdaLN-Zero block: masked self-attention + FFN, each as a gated residual (zero-init gate)."""

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float,
                 rope: RoPE | None):
        super().__init__()
        self.attn_mod = AdaLNZero(d_model)
        self.attn = MaskedMHSA(d_model, num_heads, dropout, rope)
        self.ffn_mod = AdaLNZero(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_mod.gate_out(self.attn(self.attn_mod.modulate(x), attn_mask))
        x = x + self.ffn_mod.gate_out(self.ffn(self.ffn_mod.modulate(x)))
        return x


class TemporalStream(nn.Module):
    """One masked stream (``forward`` reactive or ``backward`` anticipatory)."""

    def __init__(self, config: dict, direction: str):
        super().__init__()
        m = config["model"]
        d = int(m["d_model"])
        heads = int(m["attention_heads"])
        num_layers = int(m["num_encoder_layers"])
        dropout = float(m["dropout"])
        ffn_mult = int(m.get("ffn_mult", 4))
        self.direction = direction
        rope = RoPE(d // heads)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(d, heads, ffn_mult, dropout, rope)
            for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, frames_valid: torch.Tensor) -> torch.Tensor:
        mask = _temporal_mask(self.direction, frames_valid)
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class BidirectionalTemporal(nn.Module):
    """Temporal transformer over the gated-pool sequence.

    ``dual`` mode (default): separate reactive (forward-causal) + anticipatory (backward-causal)
    streams. ``single`` mode: one non-causal bidirectional stream whose output is returned for both
    slots, so the downstream ``cat(reactive, anticipatory)`` keeps the ``(B, T, 2d)`` interface.
    """

    def __init__(self, config: dict, mode: str = "dual"):
        super().__init__()
        self.mode = mode
        if mode == "single":
            self.stream = TemporalStream(config, "bidirectional")
        else:
            self.reactive = TemporalStream(config, "forward")
            self.anticipatory = TemporalStream(config, "backward")

    def forward(self, x: torch.Tensor, frames_valid: torch.Tensor):
        """``x: (B, T, d)`` -> ``(reactive (B, T, d), anticipatory (B, T, d))``."""
        if self.mode == "single":
            y = self.stream(x, frames_valid)
            return y, y
        return self.reactive(x, frames_valid), self.anticipatory(x, frames_valid)
