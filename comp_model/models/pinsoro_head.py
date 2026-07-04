"""PInSoRo categorical head.

Parallel decode branch from the dyadic context sequence ``X_cond (B, W, 2d)``: predicts a horizon
of social-engagement (5-class) and task-engagement (4-class) logits. Runs in parallel to the
continuous diffusion path and is invoked only on categorical (PInSoRo) batches.

A per-frame temporal-query decoder (``models/query_decoder.py``) carries K learnable queries — one
per horizon frame — that cross-attend to the W context frames: query *k* reads the context for the
+k-frame prediction. Two per-frame linear projections decode the shared ``(B, K, d)`` frame tokens
into social and task logits. String labels are mapped to integer class indices in the loader via
``registry.SOCIAL_CLASSES`` / ``registry.TASK_CLASSES``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.registry import PINSORO_SOCIAL_CLASSES, PINSORO_TASK_CLASSES
from .query_decoder import TemporalQueryDecoder


class PInSoROHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = int(config["model"]["d_model"])
        heads = int(config["model"]["attention_heads"])
        dropout = float(config["model"].get("dropout", 0.0))
        ffn_mult = int(config["model"].get("ffn_mult", 4))
        layers = int(config["model"].get("head_decoder_layers", 2))
        self.K = int(config["window"]["K"])
        self.social_classes = PINSORO_SOCIAL_CLASSES
        self.task_classes = PINSORO_TASK_CLASSES
        # Shared per-frame decoder body; two per-frame linear heads decode each frame token.
        self.decoder = TemporalQueryDecoder(
            2 * d, d, self.K, heads, layers, ffn_mult, dropout)   # X_cond (B,W,2d) -> (B,K,d)
        self.social = nn.Linear(d, self.social_classes)
        self.task = nn.Linear(d, self.task_classes)

    def forward(self, X_cond: torch.Tensor, frames_valid: torch.Tensor):
        """``X_cond: (B, W, 2d)``; ``frames_valid: (B, W)`` ->
        ``(social (B, K, 5), task (B, K, 4))`` logits."""
        h = self.decoder(X_cond, frames_valid)              # (B, K, d) — one token per horizon frame
        return self.social(h), self.task(h)
