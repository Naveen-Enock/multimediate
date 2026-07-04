"""Per-frame temporal-query decoder for the categorical / regression heads.

Carries K learnable temporal queries (one per horizon frame) that cross-attend to the full context
sequence ``X_cond (B, W, 2d)``: query *k* reads context for the +k-frame prediction, so the +1-frame
and +K-frame predictions stay distinguishable.

Each block is a standard Pre-LN transformer decoder layer: self-attention across the K queries,
cross-attention into the projected context, then an FFN. The output ``(B, K, d)`` is decoded per
frame by the owning head (social/task logits, or a regression scalar).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DecoderBlock(nn.Module):
    """Pre-LN decoder layer: self-attn over K queries -> cross-attn to context -> FFN."""

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_key_pad: torch.Tensor | None) -> torch.Tensor:
        """``x: (B, K, d)`` queries; ``ctx: (B, W, d)`` K/V; ``ctx_key_pad: (B, W)`` True=ignore."""
        h = self.norm_sa(x)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa
        h = self.norm_ca(x)
        ca, _ = self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_key_pad, need_weights=False)
        x = x + ca
        return x + self.ffn(self.norm_ff(x))


class TemporalQueryDecoder(nn.Module):
    """K learnable temporal queries cross-attending to ``X_cond`` -> per-frame ``(B, K, d)``."""

    def __init__(self, in_dim: int, d_model: int, K: int, num_heads: int,
                 num_layers: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, K, d_model) * d_model ** -0.5)
        self.ctx_proj = nn.Linear(in_dim, d_model)               # X_cond (B,W,2d) -> (B,W,d)
        self.blocks = nn.ModuleList([
            _DecoderBlock(d_model, num_heads, ffn_mult, dropout) for _ in range(num_layers)])

    def forward(self, X_cond: torch.Tensor, frames_valid: torch.Tensor) -> torch.Tensor:
        """``X_cond: (B, W, in_dim)``; ``frames_valid: (B, W)`` -> ``(B, K, d)``."""
        B = X_cond.shape[0]
        x = self.queries.expand(B, -1, -1)                       # (B, K, d)
        ctx = self.ctx_proj(X_cond)                              # (B, W, d)
        key_pad = frames_valid <= 0.5                            # (B, W) True = ignore
        key_pad = key_pad.clone()
        key_pad[:, 0] = False                                    # keep >=1 key -> no NaN row
        for blk in self.blocks:
            x = blk(x, ctx, key_pad)
        return x
