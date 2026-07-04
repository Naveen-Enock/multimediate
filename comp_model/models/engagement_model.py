"""EngagementModel — all-nodes trunk wrapper (predicts engagement for everyone in the frame).

Every batch carries all nodes of a session window: ``nodes {m: (B, N, W, D_m)}`` padded to
``N = MAX_PARTNERS + 1`` with ``node_mask (B, N)`` (real nodes) and ``is_target (B, N)`` (scored).

The per-node trunk — ModalityProjection -> UnifiedBankFusion -> GatedModalityPool -> DomainFiLM #1
-> reactive + anticipatory temporal streams — is node-independent, so it runs once over the
flattened ``(B*N)`` axis. Only the interpersonal tail is target-specific: each node cross-attends
to all *other* nodes via the shared ``GraphCrossAttention`` in an all-pairs pass (self masked out),
then DomainFiLM #2, then the parallel heads. One forward yields predictions for every person.

The reactive and anticipatory streams are concatenated along the feature axis into the per-node
context sequence ``X_cond (B*N, W, 2d)`` (time dimension unchanged); heads run over ``(B*N)`` and
outputs are reshaped to ``(B, N, …)``. Each stage is behind a config flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.registry import MODALITY_ORDER
from .bidir_transformer import BidirectionalTemporal
from .diffusion_transformer_head import DiffusionTransformerHead
from .domain_adaptation import DomainEmbedding, DomainFiLM
from .fusion import UnifiedBankFusion
from .graph_attention import GraphCrossAttention
from .modality_pool import GatedModalityPool
from .pinsoro_head import PInSoROHead
from .projection import ModalityProjection
from .regression_head import RegressionHead


class _GradScale(torch.autograd.Function):
    """Identity forward; scales the gradient by a constant on the way back."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradScale.apply(x, scale)


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
        self.use_diffusion = bool(config["model"].get("use_diffusion", True))
        self.use_domain_film = bool(config["model"].get("use_domain_film", True))
        # Scale on the categorical head's gradient into the shared trunk (1.0 = off).
        self.cat_trunk_grad_scale = float(
            (config.get("pinsoro", {}) or {}).get("trunk_grad_scale", 1.0))
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
        if self.use_diffusion:
            self.diffusion = DiffusionTransformerHead(config)
        else:
            self.head = RegressionHead(config)
        self.pinsoro_head = PInSoROHead(config)

    def unfreeze_all_for_categorical(self) -> tuple[int, int]:
        """Phase 2: train every weight except the continuous (diffusion/regression) head.

        Returns ``(trainable_params, total_params)``.
        """
        for p in self.parameters():
            p.requires_grad = True
        cont_head = self.diffusion if self.use_diffusion else self.head
        for p in cont_head.parameters():
            p.requires_grad = False
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_train, n_total

    def freeze_categorical_head(self) -> None:
        """Phase 1: freeze the PInSoRo classification head (unused by the continuous path)."""
        for p in self.pinsoro_head.parameters():
            p.requires_grad = False

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
        ``framing`` (camera perspective differs by node — e.g. the PInSoRo env scene view vs the
        children) and the active-sensor multi-hot ``node_present`` are per node. Dataset identity
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
        # Cast to fp32 (categorical batches arrive as fp16; no-op for the fp32 continuous path).
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
        K = self.K
        out = {}
        # Heads run over the flattened (B*N) node axis; outputs reshaped back to (B, N, …).
        # Routing is by the (homogeneous) batch label_kind.
        kinds = batch.get("label_kind")
        kind = kinds[0] if isinstance(kinds, list) and kinds else "continuous"
        if kind == "categorical":
            # Soften the categorical gradient flowing back into the shared trunk.
            X_cat = (grad_scale(X_cond, self.cat_trunk_grad_scale)
                     if self.cat_trunk_grad_scale != 1.0 else X_cond)
            soc, tsk = self.pinsoro_head(X_cat, fv)
            out["social_logits"] = soc.reshape(B, N, K, soc.shape[-1])   # (B, N, K, 5)
            out["task_logits"] = tsk.reshape(B, N, K, tsk.shape[-1])     # (B, N, K, 4)
        elif self.use_diffusion:
            # Per-dataset target standardization stats, one (mean, std) per node (B*N, 1).
            y_mean, y_std = self.diffusion.dataset_stats(batch["dataset_id"], N)
            if self.training:
                y0 = batch["target_chunk"].reshape(B * N, K)
                dd = self.diffusion.noise_pred(X_cond, fv, y0, y_mean, y_std)
                out["v_pred"] = dd["v_pred"].reshape(B, N, K)         # predicted velocity
                out["v_target"] = dd["v"].reshape(B, N, K)            # true velocity target
                out["y_t"] = dd["y_t"].reshape(B, N, K)
                out["alpha_bar_t"] = dd["alpha_bar_t"].reshape(B, N, 1)
                out["y_mean"] = y_mean.reshape(B, N, 1)               # per-dataset un-std stats
                out["y_std"] = y_std.reshape(B, N, 1)
            else:
                out["pred"] = self.diffusion.sample(
                    X_cond, fv, y_mean, y_std).reshape(B, N, K)      # (B, N, K)
        else:
            out["pred"] = self.head(X_cond, fv).reshape(B, N, K)         # (B, N, K)
        return out
