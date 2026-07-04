#!/usr/bin/env python3
"""Validation reporting — per-domain CCC + average + worst; PInSoRo kappa + histograms.

Continuous datasets: overlapping windows are merged per frame with temporal ensembling
(``training/ensemble.py``) before CCC. PInSoRo: predicted logits are ensembled per frame before
argmax, then unweighted Cohen's kappa (social + task + mean); ground-truth and predicted class
histograms are reported as collapse checks.
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

from comp_model.config import load_config
from comp_model.data import registry
from comp_model.data.build import build_loader
from comp_model.data.normalize import Normalizer, empty as empty_normalizer
from comp_model.metrics import ccc, cohen_kappa
from comp_model.models.engagement_model import EngagementModel
from comp_model.training.ensemble import (
    ensemble_continuous, ensemble_logits, frame_coverage, subsample_forward_fill)
from comp_model.training.trainer import move_to_device


def _eval_workers(cfg) -> int:
    """DataLoader workers for validation loaders.

    Defaults to ``data.eval_workers`` if set, else the training ``data.num_workers``. Validation is
    forward-only (no grad), so parallel loading hides disk read + decode latency behind the GPU work
    (diffusion sampling for continuous, a single forward for PInSoRo). Workers only touch CPU/numpy,
    so they are safe to spawn from the rank0 validation pass during DDP training.
    """
    data = cfg["data"]
    return int(data.get("eval_workers", data.get("num_workers", 0)))


def _scatter_labels(starts, chunks, masks, length):
    """Reconstruct a per-frame integer label sequence (-1 where uncovered/invalid)."""
    out = np.full(length, -1, dtype=np.int64)
    for t, ch, mk in zip(starts, chunks, masks):
        t = int(t)
        k = min(len(ch), length - t)
        for j in range(k):
            if mk[j] > 0.5:
                out[t + j] = int(ch[j])
    return out


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


@torch.no_grad()
def evaluate_pinsoro_kappa(model, cfg, normalizer, device, split: str = "val",
                           prior_target: dict | None = None):
    """Unweighted Cohen's kappa on a PInSoRo split: ensemble logits per frame, then argmax.

    ``split`` selects the logical split — ``val_cc`` / ``val_cr`` isolate the child-child /
    child-robot interaction types so kappa can be reported per sub-split.
    Returns ``{"social", "task"}`` (no aggregate mean) or None if the head/loader is unavailable.

    Per-window logits are forward-filled from their 1 Hz subsample (the granularity the head is
    trained at) back to 25 fps, then ensembled across overlapping windows (horizon decay) and
    argmaxed. By default logits are read **raw** — the head is calibrated entirely at train time by
    the per-domain logit-adjustment margin (``train_logit_adjust_tau_{cc,cr}``).

    ``prior_target`` optionally applies the submission's eval-time prior correction (``submit.py
    --prior-inject ...=exact``): a dict ``{"social": π, "task": π}`` of target class marginals. For
    each named axis a single global per-class logit bias (Saerens fixed-point, ``submit.fit_marginal_bias``)
    is fit so the argmax marginal over the split matches ``π``, then added before argmax.
    """
    _, loader = build_loader(cfg, ["pinsoro"], [split], train=False,
                             normalizer=normalizer, num_workers=_eval_workers(cfg))
    if loader is None:
        return None
    model.eval()
    kd = float(cfg.get("ensemble", {}).get("decay_kappa", 0.02))
    st = max(1, int(cfg.get("pinsoro", {}).get("pred_stride", 25)))
    groups = {}
    for batch in loader:
        sess, roles, starts = batch["session_id"], batch["node_roles"], batch["window_start"]
        out = model(move_to_device(batch, device))
        sl = out["social_logits"].float().cpu().numpy()    # (B, N, K, 5)
        tl = out["task_logits"].float().cpu().numpy()       # (B, N, K, 4)
        sgt = batch["target_chunk"]["social"].numpy()
        tgt = batch["target_chunk"]["task"].numpy()
        sm = batch["valid_mask"]["social"].numpy()
        tm = batch["valid_mask"]["task"].numpy()
        is_t = batch["is_target"].numpy()                   # (B, N)
        Bsz, Nn = is_t.shape
        for i in range(Bsz):
            for nd in range(Nn):
                if is_t[i, nd] < 0.5:                       # skip padded / partner-only nodes
                    continue
                g = groups.setdefault((sess[i], roles[i][nd]), {
                    "starts": [], "sl": [], "tl": [], "sgt": [], "tgt": [], "sm": [], "tm": []})
                g["starts"].append(int(starts[i]))
                g["sl"].append(subsample_forward_fill(sl[i, nd], st))
                g["tl"].append(subsample_forward_fill(tl[i, nd], st))
                g["sgt"].append(sgt[i, nd]); g["tgt"].append(tgt[i, nd])
                g["sm"].append(sm[i, nd]); g["tm"].append(tm[i, nd])

    s_true, t_true = [], []
    s_logits, t_logits = [], []          # per-session ensembled per-frame logits
    for g in groups.values():
        K = g["sl"][0].shape[0]
        length = max(g["starts"]) + K
        s_logits.append(ensemble_logits(g["starts"], g["sl"], length, kd, g["sm"]))
        t_logits.append(ensemble_logits(g["starts"], g["tl"], length, kd, g["tm"]))
        s_true.append(_scatter_labels(g["starts"], g["sgt"], g["sm"], length))
        t_true.append(_scatter_labels(g["starts"], g["tgt"], g["tm"], length))

    from comp_model.data.registry import PINSORO_SOCIAL_CLASSES, PINSORO_TASK_CLASSES
    if not s_true:
        return None

    # Optional submission-time prior correction (matches submit.py --prior-inject =exact): fit one
    # global per-class logit bias so the argmax marginal over the split matches the target, then add.
    if prior_target:
        from comp_model.eval.submit import fit_marginal_bias
        if prior_target.get("social") is not None:
            b, _ = fit_marginal_bias(s_logits, np.asarray(prior_target["social"], dtype=np.float64))
            s_logits = [x + b for x in s_logits]
        if prior_target.get("task") is not None:
            b, _ = fit_marginal_bias(t_logits, np.asarray(prior_target["task"], dtype=np.float64))
            t_logits = [x + b for x in t_logits]
    s_pred = [x.argmax(-1) for x in s_logits]
    t_pred = [x.argmax(-1) for x in t_logits]
    sk = cohen_kappa(np.concatenate(s_true), np.concatenate(s_pred),
                     num_classes=PINSORO_SOCIAL_CLASSES)
    tk = cohen_kappa(np.concatenate(t_true), np.concatenate(t_pred),
                     num_classes=PINSORO_TASK_CLASSES)
    return {"social": sk, "task": tk}


@torch.no_grad()
def pinsoro_histogram(cfg, normalizer):
    """Ground-truth social/task class histograms over PInSoRo val (collapse scaffold)."""
    _, loader = build_loader(cfg, ["pinsoro"], ["val"], train=False,
                             normalizer=normalizer, num_workers=_eval_workers(cfg))
    if loader is None:
        return None
    from comp_model.data.registry import PINSORO_SOCIAL_CLASSES, PINSORO_TASK_CLASSES
    soc = np.zeros(PINSORO_SOCIAL_CLASSES, dtype=np.int64)
    tsk = np.zeros(PINSORO_TASK_CLASSES, dtype=np.int64)
    for batch in loader:
        s = batch["target_chunk"]["social"].numpy().reshape(-1)
        t = batch["target_chunk"]["task"].numpy().reshape(-1)
        for c in s[s >= 0]:
            soc[c] += 1
        for c in t[t >= 0]:
            tsk[c] += 1
    return {"social": soc, "task": tsk}


@torch.no_grad()
def pinsoro_pred_histogram(model, cfg, normalizer, device):
    """Predicted social/task class histograms over PInSoRo val (collapse check)."""
    _, loader = build_loader(cfg, ["pinsoro"], ["val"], train=False,
                             normalizer=normalizer, num_workers=_eval_workers(cfg))
    if loader is None:
        return None
    model.eval()
    from comp_model.data.registry import PINSORO_SOCIAL_CLASSES, PINSORO_TASK_CLASSES
    soc = np.zeros(PINSORO_SOCIAL_CLASSES, dtype=np.int64)
    tsk = np.zeros(PINSORO_TASK_CLASSES, dtype=np.int64)
    for batch in loader:
        sm = batch["valid_mask"]["social"].numpy().reshape(-1).astype(bool)
        tm = batch["valid_mask"]["task"].numpy().reshape(-1).astype(bool)
        out = model(move_to_device(batch, device))
        sp = out["social_logits"].argmax(-1).cpu().numpy().reshape(-1)
        tp = out["task_logits"].argmax(-1).cpu().numpy().reshape(-1)
        for c in sp[sm]:
            soc[c] += 1
        for c in tp[tm]:
            tsk[c] += 1
    return {"social": soc, "task": tsk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="comp_model/configs/default.yaml")
    ap.add_argument("--checkpoint", default="",
                    help="path to a checkpoint, e.g. checkpoints/<run_name>/epoch_030.pt "
                         "(empty / missing -> evaluate the untrained model)")
    args = ap.parse_args()

    cfg = load_config(args.config)
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

    # PInSoRo kappa, reported per interaction type (cc = child-child, cr = child-robot).
    print("\n=== PInSoRo unweighted Cohen's kappa (per sub-split) ===")
    for tag in ("cc", "cr"):
        split = f"val_{tag}"
        kap = evaluate_pinsoro_kappa(model, cfg, normalizer, device, split=split)
        if kap is not None:
            print(f"  {tag} ({split}):  social = {kap['social']:.4f}   task = {kap['task']:.4f}")

    hist = pinsoro_histogram(cfg, normalizer)
    if hist is not None:
        print("\n=== PInSoRo GT class histogram ===")
        print(f"  social: {hist['social'].tolist()}")
        print(f"  task:   {hist['task'].tolist()}")

    pred_hist = pinsoro_pred_histogram(model, cfg, normalizer, device)
    if pred_hist is not None:
        print("\n=== PInSoRo predicted class histogram (collapse check) ===")
        print(f"  social: {pred_hist['social'].tolist()}")
        print(f"  task:   {pred_hist['task'].tolist()}")


if __name__ == "__main__":
    main()
