"""EngagementModel — all-nodes trunk wrapper (predicts engagement for everyone in the frame).

Every batch carries all nodes of a session window: ``nodes {m: (B, N, W, D_m)}`` padded to
``N = MAX_PARTNERS + 1`` with ``node_mask (B, N)`` (real nodes) and ``is_target (B, N)`` (scored).

The per-node trunk — ModalityProjection -> UnifiedBankFusion -> GatedModalityPool -> DomainFiLM #1
-> reactive + anticipatory temporal streams — is node-independent, so it runs once over the
flattened ``(B*N)`` axis. Only the interpersonal tail is target-specific: each node cross-attends
to all *other* nodes via the shared ``GraphCrossAttention`` in an all-pairs pass (self masked out),
then DomainFiLM #2, then the regression head. One forward yields predictions for every person.

The reactive and anticipatory streams are concatenated along the feature axis into the per-node
context sequence ``X_cond (B*N, W, 2d)`` (time dimension unchanged); the head runs over ``(B*N)``
and its output is reshaped to ``(B, N, K)``. Each stage is behind a config flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.registry import MODALITY_ORDER
from .bidir_transformer import BidirectionalTemporal
from .domain_adaptation import DomainEmbedding, DomainFiLM
from .fusion import UnifiedBankFusion
from .graph_attention import GraphCrossAttention
from .modality_pool import GatedModalityPool
from .projection import ModalityProjection
from .regression_head import RegressionHead


class EngagementModel(nn.Module):
    def __init__(self, config: dict, modality_dims: dict | None = None):
        super().__init__()
        d = int(config["model"]["d_model"])
        K = int(config["window"]["K"])
        self.d_model = d
        self.K = K
        self.use_fusion = bool(config["model"].get("use_fusion", True))
        self.use_temporal = bool(config["model"].get("use_temporal", True))
        self.temporal_mode = str(config["model"].get("temporal_mode", "dual"))
        self.use_graph = bool(config["model"].get("use_graph", True))
        self.use_domain_film = bool(config["model"].get("use_domain_film", True))
        self.projection = ModalityProjection(config, modality_dims=modality_dims)
        if self.use_fusion:
            self.fusion = UnifiedBankFusion(config)
            self.pool = GatedModalityPool(config)
        if self.use_temporal:
            # Runs over the pooled sequence (gated pool when use_fusion, else the mean fallback).
            self.temporal = BidirectionalTemporal(config, mode=self.temporal_mode)
        if self.use_graph:
            self.graph = GraphCrossAttention(config)
        if self.use_domain_film:
            self.domain_emb = DomainEmbedding(d)
            self.film_pool = DomainFiLM(d)       # after GatedModalityPool (per node)
            self.film_graph = DomainFiLM(d)      # after GraphCrossAttention (per node)
        self.head = RegressionHead(config)

    def _node_streams(self, node: dict, present: torch.Tensor,
                      frames_valid: torch.Tensor, domain_cond: torch.Tensor | None = None):
        """One flattened batch of nodes -> (reactive, anticipatory) ``(N', W, d)`` each.

        ``node[m]: (N', W, D_m)``, ``present: (N', M)``, ``frames_valid: (N', W)`` where ``N' = B*N``.
        ``domain_cond (N', d)`` (optional) FiLM-modulates the pooled sequence (the first
        domain-adaptation injection point, after the modality pool).
        """
        x = self.projection(node)                          # (N', W, M, d)
        N_, W, M, d = x.shape
        if self.use_fusion:
            x = self.fusion(x, present)                    # (N', W, M, d)
            seq = self.pool(x, present)                    # (N', W, d)
        else:
            # Fallback: presence-weighted mean over M (keep T) -> (N', W, d).
            pres = present.view(N_, 1, M, 1)
            seq = (x * pres).sum(dim=2) / pres.sum(dim=2).clamp_min(1.0)
        if self.use_domain_film and domain_cond is not None:
            seq = self.film_pool(seq, domain_cond)         # FiLM #1 (after modality pool)
        if self.use_temporal:
            return self.temporal(seq, frames_valid)        # (N', W, d), (N', W, d)
        return seq, seq

    def _node_domain_cond(self, batch: dict) -> torch.Tensor | None:
        """Per-node domain conditioning ``(B*N, d)``.

        ``count / kind / language`` are session-shared (broadcast over nodes); ``role`` and
        ``framing`` (camera perspective differs by node) and the active-sensor multi-hot ``node_present`` are per node. Dataset identity
        is not conditioned on.
        """
        if not self.use_domain_film or "framing_idx" not in batch:
            return None
        role = batch["role_idx"]                           # (B, N)
        B, N = role.shape
        pc = batch["partner_count_idx"].view(B, 1).expand(B, N).reshape(B * N)
        lk = batch["label_kind_idx"].view(B, 1).expand(B, N).reshape(B * N)
        lang = batch["language_idx"].view(B, 1).expand(B, N).reshape(B * N)
        fr = batch["framing_idx"].reshape(B * N)           # (B, N) per node
        sensor = batch["node_present"].reshape(B * N, -1)  # (B*N, M) active-sensor multi-hot
        return self.domain_emb(role.reshape(B * N), pc, lk, fr, lang, sensor)

    def _all_pairs_graph(self, stream: torch.Tensor, node_mask: torch.Tensor,
                         nfv: torch.Tensor) -> torch.Tensor:
        """Each node attends to all *other* nodes via the shared graph module.

        ``stream (B, N, W, d)``; ``node_mask (B, N)``; ``nfv (B, N, W)`` -> ``(B, N, W, d)``.
        For target ``i`` the partners are the other ``N-1`` nodes (gathered), with padded/absent
        nodes masked by ``node_mask``; a node never attends to itself.
        """
        B, N, W, d = stream.shape
        if N == 1:
            return stream
        idx = torch.arange(N, device=stream.device)
        partner_idx = torch.stack([torch.cat([idx[:i], idx[i + 1:]]) for i in range(N)])  # (N, P)
        P = N - 1
        partners = stream[:, partner_idx]                  # (B, N, P, W, d)
        pmask = node_mask[:, partner_idx]                  # (B, N, P)
        pfv = nfv[:, partner_idx]                          # (B, N, P, W)
        out = self.graph(
            stream.reshape(B * N, W, d),
            partners.reshape(B * N, P, W, d),
            pmask.reshape(B * N, P),
            pfv.reshape(B * N, P, W))
        return out.reshape(B, N, W, d)

    def encode(self, batch: dict):
        """All-nodes trunk + all-pairs graph -> per-node context sequence.

        Returns ``(X_cond (B*N, W, 2d), frames_valid (B*N, W), B, N)``: reactive and anticipatory
        streams concatenated along the feature axis so the time dimension stays at W.
        """
        nodes = batch["nodes"]                             # {m: (B, N, W, D_m)}
        present = batch["node_present"]                    # (B, N, M)
        nfv = batch["node_frames_valid"]                   # (B, N, W)
        node_mask = batch["node_mask"]                     # (B, N)
        B, N = node_mask.shape
        W = nfv.shape[-1]
        # Cast to fp32 (precomputed features arrive as fp16).
        flat_nodes = {m: nodes[m].reshape(B * N, W, nodes[m].shape[-1]).float()
                      for m in MODALITY_ORDER}
        flat_present = present.reshape(B * N, present.shape[-1])
        flat_fv = nfv.reshape(B * N, W)
        dcond = self._node_domain_cond(batch)              # (B*N, d) or None
        react, antic = self._node_streams(flat_nodes, flat_present, flat_fv, dcond)  # (B*N, W, d)
        d = react.shape[-1]
        if self.use_graph:
            react = self._all_pairs_graph(react.reshape(B, N, W, d), node_mask, nfv)
            antic = self._all_pairs_graph(antic.reshape(B, N, W, d), node_mask, nfv)
            react = react.reshape(B * N, W, d)
            antic = antic.reshape(B * N, W, d)
        if self.use_domain_film and dcond is not None:
            react = self.film_graph(react, dcond)          # FiLM #2 (after graph cross-attn)
            antic = self.film_graph(antic, dcond)
        X_cond = torch.cat([react, antic], dim=-1)         # (B*N, W, 2d)
        return X_cond, flat_fv, B, N

    def forward(self, batch: dict) -> dict:
        X_cond, fv, B, N = self.encode(batch)
        # The head runs over the flattened (B*N) node axis; output reshaped back to (B, N, K).
        return {"pred": self.head(X_cond, fv).reshape(B, N, self.K)}
