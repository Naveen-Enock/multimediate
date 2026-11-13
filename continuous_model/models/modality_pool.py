"""Gated Attention Pool over the modality axis.

Collapses the fused modality tokens ``(B, T, M, d)`` into one per-timestep vector ``(B, T, d)``
by dynamically weighting modalities per frame (e.g. audio vs pose). Learned attention scores over
the M tokens (absent modalities masked to ``-inf`` before the softmax), an optional per-modality
sigmoid value gate, then a weighted sum over M. Collapses M while keeping the full time axis T.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GatedModalityPool(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = int(config["model"]["d_model"])
        self.scale = 1.0 / math.sqrt(d)
        self.score = nn.Linear(d, 1)
        if bool(config["model"].get("pool_value_gate", True)):
            self.gate = nn.Linear(d, d)
            # Start near pass-through (sigmoid(2)≈0.88).
            nn.init.constant_(self.gate.bias, 2.0)
        else:
            self.gate = None

    def forward(self, x: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """``x: (B, T, M, d)`` + ``presence: (B, M)`` -> ``(B, T, d)``."""
        logits = self.score(x).squeeze(-1) * self.scale            # (B, T, M)
        absent = presence == 0                                     # (B, M)
        # A fully-absent node (padded partner) would mask every modality -> all -inf -> softmax NaN.
        # Keep all modalities for such rows so they stay finite.
        absent = absent & ~absent.all(dim=1, keepdim=True)
        logits = logits.masked_fill(absent.unsqueeze(1), float("-inf"))
        attn = torch.softmax(logits, dim=2)                        # (B, T, M); softmax over M
        xg = torch.sigmoid(self.gate(x)) * x if self.gate is not None else x
        return (attn.unsqueeze(-1) * xg).sum(dim=2)                # (B, T, d)
