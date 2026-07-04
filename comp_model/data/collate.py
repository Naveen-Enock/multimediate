"""Batch collation with variable-node padding (all-nodes session windows).

Every sample carries all nodes of one session window. Collate pads the node axis to the largest
node count in the batch (zero nodes + ``node_mask = 0``), capped at ``MAX_PARTNERS + 1``, so the
per-node trunk and the all-pairs graph layer run batched without trailing all-padded slots. NoXi
batches are N=2, PInSoRo N=3. ``is_target`` marks the scored nodes; ``node_mask`` marks the real
(non-padded) nodes. Batches are assumed homogeneous in ``label_kind`` (the trainer keeps
continuous and categorical datasets in separate loaders).
"""

from __future__ import annotations

import torch

from .registry import MODALITY_ORDER, MODALITY_DIMS, MAX_PARTNERS


def make_collate(max_partners: int = MAX_PARTNERS, window_W: int = 512,
                 modality_dims: dict | None = None):
    """``modality_dims`` overrides ``MODALITY_DIMS`` for pre-allocation — pass
    ``normalizer.effective_dims`` so tensors reflect channel selection."""
    _dims = modality_dims if modality_dims is not None else MODALITY_DIMS
    N_cap = max_partners + 1                    # topological max interactive participants

    def collate(batch: list) -> dict:
        B = len(batch)
        kind = batch[0]["label_kind"]
        N = min(max(len(b["nodes"]) for b in batch), N_cap)
        M = len(MODALITY_ORDER)
        K = batch[0]["target_chunk"].shape[-1] if kind == "continuous" \
            else batch[0]["target_chunk"]["social"].shape[-1]

        # ── node modality streams, padded to N ───────────────────────────────
        dtypes = {t.dtype for b in batch for node in b["nodes"] for t in node.values()}
        feat_dtype = torch.float16 if dtypes == {torch.float16} else torch.float32
        nodes = {m: torch.zeros(B, N, window_W, _dims[m], dtype=feat_dtype) for m in MODALITY_ORDER}
        node_present = torch.zeros(B, N, M)
        node_fv = torch.zeros(B, N, window_W)
        node_mask = torch.zeros(B, N)
        is_target = torch.zeros(B, N)
        role_idx = torch.zeros(B, N, dtype=torch.long)
        framing_idx = torch.zeros(B, N, dtype=torch.long)
        for bi, b in enumerate(batch):
            for ni, node in enumerate(b["nodes"][:N]):
                for m, t in node.items():
                    nodes[m][bi, ni] = t
                node_present[bi, ni] = b["node_present"][ni]
                node_fv[bi, ni] = b["node_frames_valid"][ni]
                node_mask[bi, ni] = 1.0
                is_target[bi, ni] = b["is_target"][ni]
                role_idx[bi, ni] = b["role_idx"][ni]
                framing_idx[bi, ni] = b["framing_idx"][ni]
        out = {
            "nodes": nodes,
            "node_present": node_present,
            "node_frames_valid": node_fv,
            "node_mask": node_mask,
            "is_target": is_target,
            "role_idx": role_idx,
            "framing_idx": framing_idx,
        }

        # ── per-node ground-truth chunks / masks, padded to N ────────────────
        if kind == "continuous":
            tc = torch.zeros(B, N, K)
            vm = torch.zeros(B, N, K)
            for bi, b in enumerate(batch):
                tc[bi, :b["target_chunk"].shape[0]] = b["target_chunk"]
                vm[bi, :b["valid_mask"].shape[0]] = b["valid_mask"]
            out["target_chunk"] = tc
            out["valid_mask"] = vm
        else:
            sc = torch.full((B, N, K), -1, dtype=torch.long)
            tk = torch.full((B, N, K), -1, dtype=torch.long)
            sm = torch.zeros(B, N, K)
            tm = torch.zeros(B, N, K)
            for bi, b in enumerate(batch):
                ns = b["target_chunk"]["social"].shape[0]
                sc[bi, :ns] = b["target_chunk"]["social"]
                tk[bi, :ns] = b["target_chunk"]["task"]
                sm[bi, :ns] = b["valid_mask"]["social"]
                tm[bi, :ns] = b["valid_mask"]["task"]
            out["target_chunk"] = {"social": sc, "task": tk}
            out["valid_mask"] = {"social": sm, "task": tm}

        # ── domain-adaptation conditioning ids (shared per session) ──────────
        for key in ("partner_count_idx", "label_kind_idx", "language_idx"):
            out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)

        # ── metadata (kept as python lists) ──────────────────────────────────
        for key in ("dataset_id", "domain", "session_id", "feature_dir", "node_roles",
                    "label_kind", "window_start"):
            out[key] = [b[key] for b in batch]
        return out

    return collate
