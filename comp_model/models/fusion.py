"""Unified Bank modality fusion.

Pre-LN AdaLN transformer running multi-head self-attention across the modality axis ``M`` at
each timestep: the M projected modality tokens are treated as a sequence and fused. Input and
output are both ``(B, T, M, d)`` — the M axis is enriched, not collapsed (the gated pool in
``models/modality_pool.py`` collapses it). No positional embedding: modality tokens are orderless.

Each block is an attention sub-layer + FFN sub-layer, both Pre-LN with affine-free LayerNorm and
static per-modality modulation (``adaln.StaticModalityModulation``), and a learnable per-modality
residual strength ``α_m`` applied as ``X_out = α_m·F(X) + (1-α_m)·X``. ``α_m`` is reused per block
and is distinct from the graph skip α_graph in ``GraphCrossAttention``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.registry import MODALITY_ORDER
from .adaln import StaticModalityModulation

# logit(0.9): per-modality block strength starts at sigmoid≈0.9.
_ALPHA_INIT = math.log(0.9 / 0.1)


class UnifiedBankBlock(nn.Module):
    """One Pre-LN AdaLN block: self-attention over M, then FFN; per-modality α_m residuals."""

    def __init__(self, d_model: int, num_modalities: int, num_heads: int,
                 ffn_mult: int, dropout: float, with_modulation: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod1 = (StaticModalityModulation(num_modalities, d_model)
                     if with_modulation else nn.Identity())
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.alpha1 = nn.Parameter(torch.full((num_modalities, 1), _ALPHA_INIT))

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod2 = (StaticModalityModulation(num_modalities, d_model)
                     if with_modulation else nn.Identity())
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.alpha2 = nn.Parameter(torch.full((num_modalities, 1), _ALPHA_INIT))

    def forward(self, xf: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``xf: (B*T, M, d)`` -> ``(B*T, M, d)`` (M is the attention sequence axis)."""
        # Attention sub-layer. alpha (M,1) broadcasts over (B*T, M, d).
        h = self.mod1(self.norm1(xf))
        z, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)
        a1 = torch.sigmoid(self.alpha1)
        xf = a1 * z + (1.0 - a1) * xf
        # FFN sub-layer.
        h = self.mod2(self.norm2(xf))
        f = self.ffn(h)
        a2 = torch.sigmoid(self.alpha2)
        xf = a2 * f + (1.0 - a2) * xf
        return xf


class UnifiedBankFusion(nn.Module):
    """Stack of ``num_encoder_layers`` Unified Bank blocks. ``(B, T, M, d)`` -> ``(B, T, M, d)``."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        d = int(m["d_model"])
        num_modalities = len(MODALITY_ORDER)
        num_layers = int(m["num_encoder_layers"])
        num_heads = int(m["attention_heads"])
        dropout = float(m["dropout"])
        ffn_mult = int(m.get("ffn_mult", 4))
        with_modulation = bool(m.get("fusion_modulation", True))
        self.mask_absent = True
        self.blocks = nn.ModuleList([
            UnifiedBankBlock(d, num_modalities, num_heads, ffn_mult, dropout, with_modulation)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, presence: torch.Tensor | None = None) -> torch.Tensor:
        """``x: (B, T, M, d)`` + optional ``presence: (B, M)`` -> ``(B, T, M, d)``."""
        B, T, M, d = x.shape
        xf = x.reshape(B * T, M, d)                     # M becomes the attention sequence axis
        kpm = None
        if self.mask_absent and presence is not None:
            # (B, M) -> (B*T, M) bool; True where a modality is absent (ignored by attention).
            absent = presence == 0                                   # (B, M)
            # A fully-absent node (padded partner slot) would mask every modality token -> a
            # fully-masked attention row -> NaN. Keep all modalities for such rows so they stay finite.
            absent = absent & ~absent.all(dim=1, keepdim=True)
            kpm = absent.unsqueeze(1).expand(B, T, M).reshape(B * T, M)
        for block in self.blocks:
            xf = block(xf, kpm)
        return xf.reshape(B, T, M, d)
