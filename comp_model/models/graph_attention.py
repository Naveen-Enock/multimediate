"""Parallel graph cross-attention — interpersonal synchronization.

Pre-LN cross-attention: both target queries and partner key/values are layer-normalised at the
entrance of the block. The aggregated output is added back as a true residual (α_graph scales the
cross-attn contribution; the target residual stream is kept at full magnitude):

    A_{T<-P} = H_T + α_graph · AGG_i CrossAttn(LN(H_T), LN(H_{P,i}), LN(H_{P,i}))

The aggregation over partners ``AGG_i`` is one of two modes (``model.graph_partner_softmax``):

  * **sum**: ``Σ_i out_i`` — every partner contributes additively.
  * **softmax** (default): a learned per-frame compatibility score gates the partners through a
    softmax over the partner axis. For a dyad (one real partner) the softmax weight is 1.

Applied separately to the reactive and anticipatory temporal streams. Partner nodes are padded to
``max_partners`` by the collate fn (zero nodes + ``partner_mask`` = 0); padded neighbors and padded
key frames are masked out. A learnable per-slot node embedding is added to each partner's keys/values
so distinct neighbor identities (e.g. the PInSoRo robot) stay separable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.registry import MAX_PARTNERS


class GraphCrossAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        d = int(m["d_model"])
        heads = int(m["attention_heads"])
        dropout = float(m["dropout"])
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.8 / 0.2)))  # sigmoid → 0.8 at init
        self.max_partners = MAX_PARTNERS
        self.partner_softmax = bool(m.get("graph_partner_softmax", True))
        self.norm_q = nn.LayerNorm(d)                            # Pre-LN on target queries
        self.norm_kv = nn.LayerNorm(d)                           # Pre-LN on partner context
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.slot_emb = nn.Embedding(self.max_partners, d)       # per-neighbor node identity
        nn.init.zeros_(self.slot_emb.weight)
        if self.partner_softmax:
            # Per-frame compatibility score between the target query and each partner's read-out.
            self.partner_score = nn.Linear(2 * d, 1)

    def forward(self, target: torch.Tensor, partners: torch.Tensor,
                partner_mask: torch.Tensor, partner_frames_valid: torch.Tensor) -> torch.Tensor:
        """target ``(B, T, d)``; partners ``(B, P, T, d)``; partner_mask ``(B, P)``;
        partner_frames_valid ``(B, P, T)`` -> ``(B, T, d)``."""
        B, P, T, d = partners.shape
        q = self.norm_q(target)                                   # Pre-LN on queries
        outs, scores = [], []
        for i in range(P):
            # Slot embedding added before normalisation so identity info is preserved.
            kv = self.norm_kv(partners[:, i] + self.slot_emb.weight[i])  # Pre-LN on KV
            # Ignore padded key frames; keep frame 0 so no row is fully masked (-> NaN).
            kpm = partner_frames_valid[:, i] <= 0.5              # (B, T) True = ignore
            kpm = kpm.clone()
            kpm[:, 0] = False
            out_i, _ = self.cross(q, kv, kv, key_padding_mask=kpm, need_weights=False)
            outs.append(out_i)                                    # (B, T, d)
            if self.partner_softmax:
                scores.append(self.partner_score(torch.cat([q, out_i], dim=-1)).squeeze(-1))  # (B,T)

        if not self.partner_softmax:
            agg = torch.zeros_like(target)
            for i in range(P):
                agg = agg + outs[i] * partner_mask[:, i].view(B, 1, 1)  # zero out padded neighbors
            return target + torch.sigmoid(self.alpha_logit) * agg

        # Normalised partner weighting: softmax learned scores over the partner axis.
        stacked = torch.stack(outs, dim=2)                       # (B, T, P, d)
        score = torch.stack(scores, dim=-1)                      # (B, T, P)
        absent = partner_mask <= 0.5                             # (B, P) True = padded neighbor
        score = score.masked_fill(absent.unsqueeze(1), float("-inf"))   # never weight padded slots
        # A node with no real partners (every slot padded) would softmax over all -inf -> NaN; make
        # the row finite (uniform) and zero its aggregate contribution below.
        no_partner = absent.all(dim=1)                           # (B,)
        score = score.masked_fill(no_partner.view(B, 1, 1), 0.0)
        weights = torch.softmax(score, dim=-1)                   # (B, T, P) over partners
        agg = (weights.unsqueeze(-1) * stacked).sum(dim=2)       # (B, T, d)
        agg = agg * (~no_partner).view(B, 1, 1)                  # no partner -> no interpersonal term
        return target + torch.sigmoid(self.alpha_logit) * agg                          # Pre-LN residual
