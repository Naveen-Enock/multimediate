"""Per-modality linear projection.

Projects each raw modality stream ``D_m -> d_model`` and stacks the streams along a new modality
axis, yielding ``(B, T, M, d)``. No pooling happens here: the M axis is collapsed later by
AdaLN-fusion + the gated modality pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.registry import MODALITY_ORDER, MODALITY_DIMS


class ModalityProjection(nn.Module):
    def __init__(self, config: dict, modality_dims: dict | None = None):
        super().__init__()
        self.d_model = int(config["model"]["d_model"])
        dims = modality_dims if modality_dims is not None else MODALITY_DIMS
        self.proj = nn.ModuleDict({
            m: nn.Linear(dims[m], self.d_model) for m in MODALITY_ORDER
        })

    def forward(self, target: dict) -> torch.Tensor:
        """``target[m]: (B, W, D_m)`` -> ``(B, W, M, d)`` stacked in MODALITY_ORDER."""
        outs = [self.proj[m](target[m]) for m in MODALITY_ORDER]
        return torch.stack(outs, dim=2)
