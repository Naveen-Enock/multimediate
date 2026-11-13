"""Domain adaptation via metadata-conditioned FiLM.

Per-node metadata is embedded into a single conditioning vector that FiLM-modulates the trunk
representation (``x -> (1 + scale) ⊙ x + shift``).

``DomainEmbedding`` turns the per-node metadata into a ``(N, d)`` conditioning vector: five learned
embeddings (``role_idx``, ``partner_count_idx``, ``label_kind_idx``, ``framing_idx``,
``language_idx``) plus a linear projection of the active-sensor multi-hot mask (``sensor_mask``, the
per-node modality present-bits) are concatenated and projected back down to ``d``, rather than
summed, so a factor whose id-space is shared across datasets (e.g. ``role_idx``, ``framing_idx``
for NoXi/NoXi-J) can't dilute a factor that is the only one varying (e.g. ``language_idx``).
``framing_idx`` encodes the spatial framing (close-up vs wide room camera vs dynamic full-body vs
scene); ``language_idx`` is the session-level spoken language. Dataset identity is not conditioned
on. ``DomainFiLM`` applies the affine modulation to a ``(B, T, d)`` sequence via a small MLP whose
final layer is zero-initialized so it starts as the identity.

Injected at two points in ``EngagementModel`` (behind ``model.use_domain_film``): right after the
``GatedModalityPool`` (per node) and right after the ``GraphCrossAttention`` (target streams).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.registry import (
    NUM_FRAMINGS, NUM_LABEL_KINDS, NUM_LANGUAGES, NUM_MODALITIES, NUM_PARTNER_COUNTS, NUM_ROLES,
)


class DomainEmbedding(nn.Module):
    """Metadata -> ``(N, d)`` domain conditioning vector.

    Concatenation of five learned id-embeddings (role / partner_count / label_kind / framing /
    language) and a linear projection of the active-sensor multi-hot mask (``sensor_mask (N, M)``,
    the per-node modality present-bits), projected back down to ``d`` by a learned ``Linear(6d, d)``.
    Concatenating (rather than summing) keeps each factor in its own sub-space until the projection,
    so factors that are constant across two datasets can't swamp the one that actually varies.
    Language is the session-level spoken language. Dataset identity is not conditioned on.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.role_emb = nn.Embedding(NUM_ROLES, d_model)
        self.partner_count_emb = nn.Embedding(NUM_PARTNER_COUNTS, d_model)
        self.label_kind_emb = nn.Embedding(NUM_LABEL_KINDS, d_model)
        self.framing_emb = nn.Embedding(NUM_FRAMINGS, d_model)
        self.language_emb = nn.Embedding(NUM_LANGUAGES, d_model)
        self.sensor_proj = nn.Linear(NUM_MODALITIES, d_model)
        self.combine = nn.Linear(6 * d_model, d_model)

    def forward(self, role_idx: torch.Tensor, partner_count_idx: torch.Tensor,
                label_kind_idx: torch.Tensor, framing_idx: torch.Tensor,
                language_idx: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        """Ids share leading shape ``(N,)``; ``sensor_mask (N, M)`` float -> conditioning ``(N, d)``."""
        parts = torch.cat([
            self.role_emb(role_idx),
            self.partner_count_emb(partner_count_idx),
            self.label_kind_emb(label_kind_idx),
            self.framing_emb(framing_idx),
            self.language_emb(language_idx),
            self.sensor_proj(sensor_mask),
        ], dim=-1)
        return self.combine(parts)


class DomainFiLM(nn.Module):
    """FiLM modulation of a ``(B, T, d)`` sequence by a ``(B, d)`` conditioning vector.

    ``x -> (1 + scale) ⊙ x + shift`` with ``(scale, shift)`` predicted from the conditioning
    vector by a two-layer MLP (``LayerNorm -> Linear -> GELU -> Linear``), so the modulation can be
    a nonlinear function of the conditioning vector rather than a single global affine map. The
    final Linear is zero-initialized so the module starts as the identity.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x: (B, T, d)``; ``cond: (B, d)`` -> ``(B, T, d)``."""
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
