#!/usr/bin/env python3
"""Validation reporting — per-domain CCC.

Overlapping windows are merged per frame with temporal ensembling (``training/ensemble.py``)
before CCC.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import yaml

from continuous_model.config import apply_overrides, load_config
from continuous_model.data import registry
from continuous_model.data.build import build_loader
from continuous_model.data.normalize import Normalizer, empty as empty_normalizer
from continuous_model.metrics import ccc
from continuous_model.models.engagement_model import EngagementModel
from continuous_model.training.ensemble import ensemble_continuous, frame_coverage
from continuous_model.training.trainer import move_to_device


def _eval_workers(cfg) -> int:
    """DataLoader workers for validation loaders.

    Defaults to ``data.eval_workers`` if set, else the training ``data.num_workers``. Validation is
    forward-only (no grad), so parallel loading hides disk read + decode latency behind the GPU work
    (a single forward per batch). Workers only touch CPU/numpy,
    so they are safe to spawn from the rank0 validation pass during DDP training.
    """
    data = cfg["data"]
    return int(data.get("eval_workers", data.get("num_workers", 0)))


@torch.no_grad()
def evaluate_continuous(model, cfg, normalizer, device, datasets, split: str = "val"):
    """Per-domain CCC with temporal ensembling (overlapping windows merged per frame)."""
    model.eval()
    kd = float(cfg.get("ensemble", {}).get("decay_kappa", 0.02))
    # group windows by (dataset, session, participant) so overlapping chunks merge per frame.
    groups = {}
    for dataset in datasets:
        _, loader = build_loader(cfg, [dataset], [split], train=False,
                                 normalizer=normalizer, num_workers=_eval_workers(cfg))
        if loader is None:
            continue
        for batch in loader:
            dom, sess = batch["domain"], batch["session_id"]
            roles, starts = batch["node_roles"], batch["window_start"]
            p = model(move_to_device(batch, device))["pred"].float().cpu().numpy()  # (B, N, K)
            t = batch["target_chunk"].float().numpy()
            m = batch["valid_mask"].float().numpy()
            is_t = batch["is_target"].numpy()                                       # (B, N)
            Bsz, Nn = is_t.shape
            for i in range(Bsz):
                for nd in range(Nn):
                    if is_t[i, nd] < 0.5:                  # skip padded / partner-only nodes
                        continue
                    g = groups.setdefault(
                        (dataset, sess[i], roles[i][nd]),
                        {"domain": dom[i], "starts": [], "preds": [], "gts": [], "masks": []})
                    g["starts"].append(int(starts[i]))
                    g["preds"].append(p[i, nd]); g["gts"].append(t[i, nd])
                    g["masks"].append(m[i, nd])

    by_p, by_t = defaultdict(list), defaultdict(list)
    for g in groups.values():
        K = len(g["preds"][0])
        length = max(g["starts"]) + K
        pe = ensemble_continuous(g["starts"], g["preds"], length, kd, g["masks"])
        te = ensemble_continuous(g["starts"], g["gts"], length, kd, g["masks"])
        cov = frame_coverage(g["starts"], length, K, g["masks"])
        by_p[g["domain"]].append(pe[cov])
        by_t[g["domain"]].append(te[cov])

    results = {}
    for dom in sorted(by_p):
        pv = np.concatenate(by_p[dom]) if by_p[dom] else np.zeros(0)
        tv = np.concatenate(by_t[dom]) if by_t[dom] else np.zeros(0)
        results[dom] = ccc(tv, pv)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="continuous_model/configs/default.yaml")
    ap.add_argument("--checkpoint", default="",
                    help="path to a checkpoint, e.g. checkpoints/<run_name>/epoch_030.pt "
                         "(empty / missing -> evaluate the untrained model)")
    ap.add_argument("--eval-stride", type=int, default=None,
                    help="override window.eval_S: slide eval windows this many frames so chunks "
                         "overlap and training/ensemble.py merges K/stride predictions per frame. "
                         "Costs (S/eval_S)x the forwards; must be < window.K to have any effect.")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, repeatable (values parsed as YAML)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    for kv in args.overrides:
        k, _, v = kv.partition("=")
        overrides[k] = yaml.safe_load(v)
    if args.eval_stride is not None:
        overrides["window.eval_S"] = int(args.eval_stride)
    if overrides:
        apply_overrides(cfg, overrides)
        print(f"overrides: {overrides}")
    es = cfg["window"].get("eval_S") or cfg["window"]["S"]
    K = cfg["window"]["K"]
    note = (f"~{max(1, K // es)} overlapping chunks/frame (temporal ensembling active)"
            if es < K else "no overlap (ensembling inert)")
    print(f"eval windowing: stride {es}, horizon K={K} -> {note}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = cfg["data"]["norm_stats"]
    normalizer = (Normalizer.load(stats) if os.path.exists(stats)
                  else empty_normalizer())

    model = EngagementModel(cfg, modality_dims=normalizer.effective_dims).to(device)
    if os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.checkpoint}")
    else:
        print(f"no checkpoint at {args.checkpoint} — evaluating untrained model")

    cont = [d for d in cfg["data"]["val_datasets"]
            if registry.label_kind(d) == "continuous"]
    results = evaluate_continuous(model, cfg, normalizer, device, cont)

    print("\n=== Continuous CCC (per-domain) ===")
    for dom, c in results.items():
        print(f"  {dom:12s} CCC = {c:.4f}")
    if results:
        print(f"  {'mean':12s} CCC = {sum(results.values()) / len(results):.4f}")


if __name__ == "__main__":
    main()
