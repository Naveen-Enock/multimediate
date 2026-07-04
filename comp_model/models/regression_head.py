"""Regression head: X_cond -> Y_t in R^K (continuous tracks when ``use_diffusion=false``).

Accepts the full dyadic context sequence ``X_cond (B, W, 2d)`` and decodes the K-frame horizon with
a per-frame temporal-query decoder (``models/query_decoder.py``, shared with ``PInSoROHead``): K
learnable queries cross-attend to the W context frames and a per-frame linear maps each query token
to its scalar prediction. Same interface as the diffusion head (X_cond + frames_valid).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .query_decoder import TemporalQueryDecoder


class RegressionHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = int(config["model"]["d_model"])
        K = int(config["window"]["K"])
        heads = int(config["model"]["attention_heads"])
        dropout = float(config["model"].get("dropout", 0.0))
        ffn_mult = int(config["model"].get("ffn_mult", 4))
        layers = int(config["model"].get("head_decoder_layers", 2))
        self.decoder = TemporalQueryDecoder(2 * d, d, K, heads, layers, ffn_mult, dropout)
        self.out = nn.Linear(d, 1)                          # per-frame scalar

    def forward(self, X_cond: torch.Tensor, frames_valid: torch.Tensor) -> torch.Tensor:
        """``X_cond: (B, W, 2d)``; ``frames_valid: (B, W)`` -> ``(B, K)``."""
        h = self.decoder(X_cond, frames_valid)              # (B, K, d)
        return self.out(h).squeeze(-1)                      # (B, K)
