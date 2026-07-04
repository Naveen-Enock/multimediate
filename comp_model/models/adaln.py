"""LayerNorm modulation helpers.

``StaticModalityModulation`` — per-modality scale/shift for the Unified Bank fusion.
``AdaLNZero`` — standard LayerNorm + zero-initialized per-channel output gate for the
reactive/anticipatory temporal blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StaticModalityModulation(nn.Module):
    """Per-modality affine ``(1 + γ_m) ⊙ x + β_m`` with no conditioning input.

    ``gamma`` and ``beta`` are learnable parameters of shape ``(M, d)``, zero-initialized so the
    modulation starts as identity. The ``(M, d)`` params broadcast against any tensor ending in
    ``(M, d)`` (e.g. ``(B, T, M, d)`` or the flattened ``(B*T, M, d)``); the M axis aligns with
    ``MODALITY_ORDER``.
    """

    def __init__(self, num_modalities: int, d_model: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_modalities, d_model))
        self.beta = nn.Parameter(torch.zeros(num_modalities, d_model))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """``x_norm: (..., M, d)`` -> same shape, per-modality scaled/shifted."""
        return (1.0 + self.gamma) * x_norm + self.beta


class AdaLNZero(nn.Module):
    """Standard LayerNorm + zero-initialized per-channel output gate for one transformer sub-layer.

    The gate starts at zero so the sub-layer is identity at init.

    Usage: ``x = x + mod.gate_out(SubLayer(mod.modulate(x)))``.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(d_model))

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def gate_out(self, y: torch.Tensor) -> torch.Tensor:
        return self.gate * y
