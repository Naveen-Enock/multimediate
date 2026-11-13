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
                 ffn_mult: int, dropout: float, with_modulation: bool,
                 gate: str = "alpha", layerscale_init: float = 1e-4):
        super().__init__()
        self.gate = gate
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod1 = (StaticModalityModulation(num_modalities, d_model)
                     if with_modulation else nn.Identity())
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        if gate == "layerscale":
            self.scale1 = nn.Parameter(torch.full((d_model,), layerscale_init))
        else:
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
        if gate == "layerscale":
            self.scale2 = nn.Parameter(torch.full((d_model,), layerscale_init))
        else:
            self.alpha2 = nn.Parameter(torch.full((num_modalities, 1), _ALPHA_INIT))

    def _residual(self, xf: torch.Tensor, delta: torch.Tensor, idx: int) -> torch.Tensor:
        """Combine the sub-layer output ``delta`` back into ``xf``.

        ``alpha``: the original per-modality convex blend ``a*delta + (1-a)*xf``, ``a = sigmoid(α_m)``
        with ``α_m`` a single scalar per modality. Measured on a trained run, these 7 scalars per
        sub-layer do not move: 0.900 -> 0.899-0.901 over 6760 steps, a relative weight change of
        0.0025 against 0.07-3.35 for every other parameter group in the same module. The per-step
        gradient is not small (|grad| ~7e1), and Adam at lr 5.5e-5 could have shifted the logit by
        ~0.37 over that many steps, so the gradient cannot be accumulating — one scalar shared across
        every (B*T, d) position receives conflicting signals and random-walks around its init. Either
        way the block cannot learn to reduce its own contribution, and it starts 90% engaged
        (``||f(x)-x||/||x|| = 1.02`` at init: a perturbation the size of the signal itself).

        ``layerscale``: per-channel residual ``xf + s⊙delta`` with ``s ∈ R^d`` initialised near zero
        (CaiT). Each channel owns its own gate, so a per-channel signal survives instead of being
        averaged into one shared scalar. The decisive property is the init: at ``s=1e-4`` the block
        is the identity (``||f(x)-x||/||x|| = 4.5e-5``), so the model *starts* at the mean-fusion
        fallback, and fusion can only grow
        where it earns its way in.
        """
        if self.gate == "layerscale":
            s = self.scale1 if idx == 1 else self.scale2
            return xf + s * delta                      # (d,) broadcasts over (B*T, M, d)
        a = torch.sigmoid(self.alpha1 if idx == 1 else self.alpha2)
        return a * delta + (1.0 - a) * xf

    def forward(self, xf: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``xf: (B*T, M, d)`` -> ``(B*T, M, d)`` (M is the attention sequence axis)."""
        # Attention sub-layer. alpha (M,1) broadcasts over (B*T, M, d).
        h = self.mod1(self.norm1(xf))
        z, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)
        xf = self._residual(xf, z, 1)
        # FFN sub-layer.
        h = self.mod2(self.norm2(xf))
        f = self.ffn(h)
        xf = self._residual(xf, f, 2)
        return xf


class UnifiedBankFusion(nn.Module):
    """Stack of ``num_encoder_layers`` Unified Bank blocks. ``(B, T, M, d)`` -> ``(B, T, M, d)``."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        d = int(m["d_model"])
        num_modalities = len(MODALITY_ORDER)
        # ``num_encoder_layers`` is shared with the temporal stack (reactive + anticipatory),
        # so it cannot set the fusion depth alone. ``fusion_num_layers`` overrides it here only;
        # null/absent = inherit. Attention over M=7 orderless tokens is a complete graph, so the
        # receptive field already saturates at one layer.
        _fl = m.get("fusion_num_layers")
        num_layers = int(m["num_encoder_layers"] if _fl is None else _fl)
        num_heads = int(m["attention_heads"])
        # ``dropout`` is shared with the temporal stack, graph attention and both heads.
        # ``fusion_dropout`` overrides it for these blocks only; null/absent = inherit. Checked
        # against None rather than falsiness so an explicit 0.0 disables dropout here.
        _fd = m.get("fusion_dropout")
        dropout = float(m["dropout"] if _fd is None else _fd)
        # ``ffn_mult`` is shared with the temporal stack and both heads, so it cannot size the
        # fusion FFN alone. ``fusion_ffn_mult`` overrides it here only; null/absent = inherit.
        _fm = m.get("fusion_ffn_mult")
        ffn_mult = int(m.get("ffn_mult", 4) if _fm is None else _fm)
        with_modulation = bool(m.get("fusion_modulation", True))
        gate = str(m.get("fusion_gate", "alpha"))
        if gate not in ("alpha", "layerscale"):
            raise ValueError(f"model.fusion_gate must be 'alpha' or 'layerscale', got {gate!r}")
        ls_init = float(m.get("fusion_layerscale_init", 1e-4))
        self.mask_absent = True
        self.blocks = nn.ModuleList([
            UnifiedBankBlock(d, num_modalities, num_heads, ffn_mult, dropout, with_modulation,
                             gate=gate, layerscale_init=ls_init)
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
