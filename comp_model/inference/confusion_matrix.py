#!/usr/bin/env python3
"""Confusion matrices for PInSoRo val from a trained checkpoint.

Runs both splits (val-cc, val-cr) and both heads (social, task) through one checkpoint and prints
per-axis confusion matrices with precision/recall/accuracy. Runs on CPU only.

Usage:
    python comp_model/inference/confusion_matrix.py \
        --config comp_model/configs/default.yaml \
        --checkpoint checkpoints/<run>/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from comp_model.config import load_config
from comp_model.data import registry
from comp_model.data.build import build_loader
from comp_model.data.normalize import Normalizer, empty as empty_normalizer
from comp_model.models.engagement_model import EngagementModel
from comp_model.training.ensemble import ensemble_logits, subsample_forward_fill
from comp_model.training.trainer import move_to_device
from comp_model.eval.submit import patch_language_embedding_to_mean


DEVICE = torch.device("cpu")

SOCIAL_NAMES = ["solitary", "onlooker", "parallel", "associative", "cooperative"]
TASK_NAMES   = ["noplay", "aimless", "goaloriented", "adultseeking"]


def _scatter_labels(starts, chunks, masks, length):
    out = np.full(length, -1, dtype=np.int64)
    for s, ch, mk in zip(starts, chunks, masks):
        for dt, (label, valid) in enumerate(zip(ch, mk)):
            t = s + dt
            if 0 <= t < length and valid > 0.5 and label >= 0:
                out[t] = int(label)
    return out


@torch.no_grad()
def run_split(model, cfg, split, heads):
    """Run inference on pinsoro/{split} for the requested heads.

    Returns per-session dicts with ensembled logits and GT label arrays.
    """
    heads = set(heads)
    want_s, want_t = "social" in heads, "task" in heads
    W = int(cfg["window"]["W"])
    K = int(cfg["window"]["K"])
    kd = float(cfg.get("ensemble", {}).get("decay_kappa", 0.02))
    st = max(1, int(cfg.get("pinsoro", {}).get("pred_stride", 25)))

    _, loader = build_loader(cfg, ["pinsoro"], [split], train=False,
                             normalizer=_normalizer, num_workers=0)
    if loader is None:
        print(f"  no sessions for pinsoro/{split}")
        return {}

    groups = {}
    for batch in loader:
        out = model(move_to_device(batch, DEVICE))
        sl = out["social_logits"].float().cpu().numpy() if want_s else None  # (B,N,K,5)
        tl = out["task_logits"].float().cpu().numpy()  if want_t else None   # (B,N,K,4)
        sgt = batch["target_chunk"]["social"].numpy() if want_s else None
        tgt = batch["target_chunk"]["task"].numpy()   if want_t else None
        sm  = batch["valid_mask"]["social"].numpy()   if want_s else None
        tm  = batch["valid_mask"]["task"].numpy()     if want_t else None
        is_t = batch["is_target"].numpy()
        sess, roles, starts = batch["session_id"], batch["node_roles"], batch["window_start"]
        B, N = is_t.shape
        for i in range(B):
            for nd in range(N):
                if is_t[i, nd] < 0.5:
                    continue
                key = (sess[i], roles[i][nd])
                g = groups.setdefault(key, {"starts": [], "sl": [], "tl": [],
                                            "sgt": [], "tgt": [], "sm": [], "tm": []})
                g["starts"].append(int(starts[i]))
                if want_s:
                    g["sl"].append(subsample_forward_fill(sl[i, nd], st))
                    g["sgt"].append(sgt[i, nd])
                    g["sm"].append(sm[i, nd])
                if want_t:
                    g["tl"].append(subsample_forward_fill(tl[i, nd], st))
                    g["tgt"].append(tgt[i, nd])
                    g["tm"].append(tm[i, nd])

    results = {}
    for key, g in groups.items():
        length = max(g["starts"]) + K
        r = {}
        if want_s:
            r["s_logit"] = ensemble_logits(g["starts"], g["sl"], length, kd, g["sm"])
            r["s_true"]  = _scatter_labels(g["starts"], g["sgt"], g["sm"], length)
        if want_t:
            r["t_logit"] = ensemble_logits(g["starts"], g["tl"], length, kd, g["tm"])
            r["t_true"]  = _scatter_labels(g["starts"], g["tgt"], g["tm"], length)
        results[key] = r
    return results


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names, title):
    n = len(class_names)
    total = cm.sum()
    col_w = max(len(c) for c in class_names) + 2
    row_label_w = max(len(c) for c in class_names) + 2
    header = " " * row_label_w + "".join(f"{c:>{col_w}}" for c in class_names)
    print(f"\n{title}")
    print(header)
    for i, row_name in enumerate(class_names):
        row = f"{row_name:<{row_label_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(n))
        print(row)

    print()
    print(f"  {'Class':<20s}  {'Precision':>10}  {'Recall':>10}  {'Accuracy':>10}  {'Support':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    precisions, recalls, accuracies = [], [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        col_sum = cm[:, i].sum()   # TP + FP
        row_sum = cm[i, :].sum()   # TP + FN
        tn = total - row_sum - col_sum + tp   # total - FN - FP - TP = TN
        prec = tp / col_sum if col_sum > 0 else 0.0
        rec  = tp / row_sum if row_sum > 0 else 0.0
        acc  = (tp + tn) / total  if total  > 0 else 0.0
        precisions.append(prec); recalls.append(rec); accuracies.append(acc)
        print(f"  {name:<20s}  {prec:>10.4f}  {rec:>10.4f}  {acc:>10.4f}  {row_sum:>10}")

    print(f"\n  {'Macro avg':<20s}  {np.mean(precisions):>10.4f}  {np.mean(recalls):>10.4f}  {np.mean(accuracies):>10.4f}")
    overall_acc = np.trace(cm) / max(total, 1)
    print(f"  {'Overall accuracy':<20s}  {overall_acc:>10.4f}  (= sum diagonal / total frames)")


def load_model(checkpoint):
    global _cfg, _normalizer
    m = EngagementModel(_cfg, modality_dims=_normalizer.effective_dims).to(DEVICE)
    m.load_state_dict(torch.load(checkpoint, map_location=DEVICE)["model"])
    m.eval()
    patch_language_embedding_to_mean(m, _cfg)
    print(f"  loaded {checkpoint}")
    return m


_cfg = None
_normalizer = None


def main():
    global _cfg, _normalizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="comp_model/configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    _cfg = load_config(args.config)
    stats = _cfg["data"]["norm_stats"]
    _normalizer = Normalizer.load(stats) if os.path.exists(stats) else empty_normalizer()

    model = load_model(args.checkpoint)

    # Accumulate GT and predictions per head across both splits.
    all_s_true, all_s_pred = [], []
    all_t_true, all_t_pred = [], []

    for split in ("val_cc", "val_cr"):
        print(f"\n[{split}]")
        results = run_split(model, _cfg, split, ("social", "task"))
        print(f"  {len(results)} (session, role) groups from pinsoro/{split}")
        for r in results.values():
            if "s_logit" in r:
                mask = r["s_true"] >= 0
                all_s_true.append(r["s_true"][mask])
                all_s_pred.append(r["s_logit"][mask].argmax(-1))
            if "t_logit" in r:
                mask = r["t_true"] >= 0
                all_t_true.append(r["t_true"][mask])
                all_t_pred.append(r["t_logit"][mask].argmax(-1))

    print("\n" + "=" * 60)

    if all_s_true:
        yt = np.concatenate(all_s_true)
        yp = np.concatenate(all_s_pred)
        cm = confusion_matrix(yt, yp, len(SOCIAL_NAMES))
        print_confusion_matrix(cm, SOCIAL_NAMES,
                               "SOCIAL ENGAGEMENT — confusion matrix (rows=GT, cols=pred)")

    if all_t_true:
        yt = np.concatenate(all_t_true)
        yp = np.concatenate(all_t_pred)
        cm = confusion_matrix(yt, yp, len(TASK_NAMES))
        print_confusion_matrix(cm, TASK_NAMES,
                               "TASK ENGAGEMENT — confusion matrix (rows=GT, cols=pred)")


if __name__ == "__main__":
    main()
