#!/usr/bin/env python3
"""Write a MultiMediate'26 challenge submission from a trained checkpoint.

Runs test-set inference and writes per-frame prediction CSVs in the official folder structure, then
(optionally) zips them. The submission tree is::

    <out>/
      noxi-base/<sess>/{expert,novice}.engagement.prediction.csv          (CCC, 25 fps, [0,1])
      noxi-additional/<sess>/{expert,novice}.engagement.prediction.csv
      noxi-j/<sess>/{expert,novice}.engagement.prediction.csv
      mpiigroupinteraction/<sess>/subjectPos{1..4}.engagement.prediction.csv
      pinsoro-cc/<sess>/{purple,yellow}.{social,task}_engagement.prediction.csv   (kappa, 30 fps)
      pinsoro-cr/<sess>/purple.{social,task}_engagement.prediction.csv

Inference mirrors ``eval/validate.py`` (temporal ensembling of overlapping windows) but writes
**absolute-frame** sequences over the *full* session: each window predicts frames
``[start+W-K, start+W)``, so chunks are placed at their true start ``start+(W-K)`` and head frames
``[0, W-K)`` (plus any interior gaps) are filled by interpolation/edge-hold over the covered frames.
Continuous sessions write the 25 fps grid directly; PInSoRo argmax indices are nearest-resampled
back to the native 30 fps annotation length and emitted as class strings. PInSoRo logits are read
raw — the head is calibrated at train time by the per-domain logit-adjustment margin
(``pinsoro.train_logit_adjust_tau_{cc,cr}``), with no eval-time prior injection by default.

Language embedding patch (always on): NoXi-additional sessions use languages absent from train+val,
whose ``language_emb`` rows never received a training gradient. Before inference we overwrite every
unseen language row with the mean of the seen rows; a no-op for every split except noxi-additional.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from comp_model.config import load_config
from comp_model.data import registry, streams
from comp_model.data.build import build_loader
from comp_model.data.normalize import Normalizer, empty as empty_normalizer
from comp_model.models.engagement_model import EngagementModel
from comp_model.data.windowing import resample_categorical_nearest
from comp_model.training.ensemble import (
    ensemble_continuous, ensemble_logits, subsample_forward_fill)
from comp_model.training.trainer import move_to_device

# Continuous (CCC) jobs: (output folder, dataset key, logical split).
CONTINUOUS_JOBS = [
    ("noxi-base",            "noxi",    "test_base"),
    ("noxi-additional",      "noxi",    "test_additional"),
    ("noxi-j",               "noxi_j",  "test"),
    ("mpiigroupinteraction", "mpii",    "test"),
]

# Categorical (kappa) jobs: (output folder, logical split). Both heads (social + task) are written.
CATEGORICAL_JOBS = [
    ("pinsoro-cc", "test_cc"),
    ("pinsoro-cr", "test_cr"),
]

SOCIAL_BY_IDX = {v: k for k, v in registry.SOCIAL_CLASSES.items()}
TASK_BY_IDX = {v: k for k, v in registry.TASK_CLASSES.items()}

_INJECT_STREAMS = {"cc-social": ("cc", "social"), "cc-task": ("cc", "task"),
                   "cr-social": ("cr", "social"), "cr-task": ("cr", "task")}


def parse_prior_inject(spec: str) -> dict:
    """Parse ``--prior-inject`` into ``{(domain, head): {"mode": str, "beta": float}}``.

    ``spec`` is comma-separated ``stream=mode`` (e.g. ``cr-social=exact,cc-task=0.3``); ``mode`` is
    ``exact`` or a float β. Empty string → no injection.
    """
    out: dict = {}
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        stream, sep, val = part.partition("=")
        stream, val = stream.strip(), val.strip().lower()
        if not sep or stream not in _INJECT_STREAMS:
            raise SystemExit(f"--prior-inject: bad entry '{part}' "
                             f"(stream must be one of {sorted(_INJECT_STREAMS)})")
        if val == "exact":
            out[_INJECT_STREAMS[stream]] = {"mode": "exact", "beta": 0.0}
        else:
            try:
                out[_INJECT_STREAMS[stream]] = {"mode": "beta", "beta": float(val)}
            except ValueError:
                raise SystemExit(f"--prior-inject: '{val}' must be 'exact' or a float β")
    return out


# ── prior correction (per stream) ─────────────────────────────────────────────
# Scored roles per interaction domain (cr's yellow is the robot, never scored).
_PRIOR_ROLES = {"cc": ("purple", "yellow"), "cr": ("purple",)}


def val_target_prior(cfg, domain: str, head: str) -> np.ndarray:
    """Empirical class prior of ``{head}`` for ``{domain}`` from the local ``val-{domain}`` labels.

    The best available estimate of the test-split marginal for a stream whose class prior shifts
    between train and val/test. ``run_categorical`` shifts the matching head's ensembled logits
    toward this prior.
    """
    classes = registry.SOCIAL_CLASSES if head == "social" else registry.TASK_CLASSES
    by_idx = SOCIAL_BY_IDX if head == "social" else TASK_BY_IDX
    root = cfg["data"]["roots"]["pinsoro"]
    counts = np.zeros(len(classes), dtype=np.float64)
    for role in _PRIOR_ROLES[domain]:
        for fp in glob.glob(os.path.join(
                root, f"val-{domain}", "*", f"{role}.{head}_engagement.annotation.csv")):
            with open(fp) as f:
                for line in f:
                    s = line.strip()
                    if s in classes:
                        counts[classes[s]] += 1
    total = counts.sum()
    if total == 0:
        raise RuntimeError(f"no val-{domain} {head} annotations under {root}/val-{domain}")
    prior = counts / total
    print(f"  {domain}-{head} target prior (val-{domain}): "
          + "  ".join(f"{by_idx[i]}={prior[i]:.3f}" for i in range(len(prior))))
    return prior


def _argmax_marginal(logits_list, n_classes: int) -> np.ndarray:
    """Class frequencies of the per-frame argmax pooled over a list of ``(T, n_classes)`` logits."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for x in logits_list:
        counts += np.bincount(x.argmax(-1), minlength=n_classes)
    return counts / max(counts.sum(), 1.0)


def fit_marginal_bias(logits_list, target, iters: int = 300, step: float = 1.0,
                      tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Per-class additive logit bias whose argmax marginal matches ``target`` (Saerens fixed-point).

    Iterates ``b ← b + step·(log π_target − log q)`` where ``q`` is the realized argmax marginal of
    ``logits + b``; warm-started from the one-shot log-ratio and returning the lowest-L1 bias seen.
    """
    eps = 1e-6
    tgt = np.clip(np.asarray(target, dtype=np.float64), eps, None)
    tgt = tgt / tgt.sum()
    log_tgt = np.log(tgt)
    n = len(tgt)
    L = np.concatenate([x.reshape(-1, n) for x in logits_list], axis=0)
    b = log_tgt - np.log(np.clip(_argmax_marginal(logits_list, n), eps, None))  # one-shot warm start
    best_b, best_d = b.copy(), np.inf
    for _ in range(iters):
        q = np.bincount((L + b).argmax(1), minlength=n).astype(np.float64) / len(L)
        d = float(np.abs(q - tgt).sum())
        if d < best_d:
            best_d, best_b = d, b.copy()
        if d < tol:
            break
        b = b + step * (log_tgt - np.log(np.clip(q, eps, None)))
    realized = np.bincount((L + best_b).argmax(1), minlength=n).astype(np.float64) / len(L)
    return best_b, realized


def _apply_prior_shift(results, key: str, spec: dict, tag: str) -> None:
    """Shift the ``key`` logits of every result in place toward ``spec['target']`` and log it.

    ``spec['mode']`` is ``"exact"`` (fit the bias so the realized argmax marginal equals the target)
    or ``"beta"`` (one-shot ``spec['beta']·(log π_target − log π_pred)``).
    """
    target = np.asarray(spec["target"], dtype=np.float64)
    n = len(target)
    logits = [r[key] for r in results]
    if spec["mode"] == "exact":
        bias, realized = fit_marginal_bias(logits, target)
        label = "exact-match"
    else:
        eps = 1e-6
        pred = np.clip(_argmax_marginal(logits, n), eps, None)
        tgt = np.clip(target, eps, None)
        tgt = tgt / tgt.sum()
        bias = float(spec["beta"]) * (np.log(tgt) - np.log(pred))
        realized = None
        label = f"β={spec['beta']}"
    for r in results:
        r[key] = r[key] + bias
    if realized is None:
        realized = _argmax_marginal([r[key] for r in results], n)
    tgt_n = target / target.sum()
    print(f"  {tag} marginal shift ({label}): "
          f"realized=[{', '.join(f'{x:.3f}' for x in realized)}] "
          f"target=[{', '.join(f'{x:.3f}' for x in tgt_n)}]")


# ── language embedding mean-centroid patch ────────────────────────────────────
def seen_languages(cfg) -> set[int]:
    """Language ids that appear in the configured train datasets/splits (got gradients)."""
    roots = cfg["data"]["roots"]
    seen: set[int] = set()
    for dataset in cfg["data"]["train_datasets"]:
        for split in cfg["data"]["train_splits"]:
            if split not in registry.splits_for(dataset):
                continue
            for ref in registry.iter_sessions(dataset, roots[dataset], split):
                seen.add(registry.language_id(ref.feature_dir, ref.dataset))
    return seen


def patch_language_embedding_to_mean(model, cfg) -> None:
    """Overwrite unseen ``language_emb`` rows with the mean of the seen (trained) rows."""
    if not getattr(model, "use_domain_film", False):
        print("  domain FiLM disabled — skipping language patch")
        return
    seen = sorted(seen_languages(cfg))
    weight = model.domain_emb.language_emb.weight
    n = weight.shape[0]
    unseen = [i for i in range(n) if i not in set(seen)]
    if not seen or not unseen:
        print(f"  language patch no-op (seen={seen}, unseen={unseen})")
        return
    by_idx = {v: k for k, v in registry.LANGUAGE_CLASSES.items()}
    with torch.no_grad():
        centroid = weight.data[seen].mean(0)
        for i in unseen:
            weight.data[i] = centroid
    print(f"  patched unseen language rows {[by_idx[i] for i in unseen]} "
          f"-> mean of seen {[by_idx[i] for i in seen]}")


# ── per-frame reconstruction helpers ──────────────────────────────────────────
def _fill_uncovered(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Fill frames with no window coverage by interpolation + edge-hold over covered frames.

    ``values``/``covered``: ``(T,)`` and ``(T,)`` bool. Head frames ``[0, W-K)`` are never predicted
    (the earliest chunk starts at frame ``W-K``); they take the first covered value. Interior gaps
    interpolate, the tail edge-holds.
    """
    idx = np.flatnonzero(covered)
    if idx.size == 0:
        return values
    if idx.size == values.shape[0]:
        return values
    return np.interp(np.arange(values.shape[0]), idx, values[idx])


def native_length(feature_dir: str, role: str) -> tuple[int, float]:
    """Native openpose ``(num_frames, sr)`` for one role (the annotation grid for that dataset)."""
    path = os.path.join(feature_dir, f"{role}.{registry.ROLE_PROBE_MODALITY}.stream")
    num, sr, _ = streams.read_stream_header(path)
    return num, sr


def write_csv_rows(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r])


# ── continuous (CCC) jobs ─────────────────────────────────────────────────────
@torch.no_grad()
def run_continuous(model, cfg, normalizer, device, dataset, split, out_dir):
    """Write ``{role}.engagement.prediction.csv`` for every scored node of a continuous test split."""
    W, K = int(cfg["window"]["W"]), int(cfg["window"]["K"])
    kd = float(cfg.get("ensemble", {}).get("decay_kappa", 0.02))
    workers = int(cfg["data"].get("eval_workers", cfg["data"].get("num_workers", 0)))
    _, loader = build_loader(cfg, [dataset], [split], train=False,
                             normalizer=normalizer, num_workers=workers)
    if loader is None:
        print(f"  no sessions for {dataset}/{split}")
        return
    feat_dirs: dict[str, str] = {}
    groups = defaultdict(lambda: {"starts": [], "preds": []})
    for batch in loader:
        pred = model(move_to_device(batch, device))["pred"].float().cpu().numpy()  # (B, N, K)
        is_t = batch["is_target"].numpy()
        roles, sess, starts = batch["node_roles"], batch["session_id"], batch["window_start"]
        feat = batch["feature_dir"]
        B, N = is_t.shape
        for i in range(B):
            for nd in range(N):
                if is_t[i, nd] < 0.5:
                    continue
                key = (sess[i], roles[i][nd])
                feat_dirs[key] = feat[i]
                g = groups[key]
                g["starts"].append(int(starts[i]) + (W - K))   # true absolute chunk start
                g["preds"].append(pred[i, nd])

    for (sess, role), g in sorted(groups.items()):
        num, sr = native_length(feat_dirs[(sess, role)], role)
        length = max(int(round(num * registry.LABEL_FPS_CONTINUOUS / sr)), 1) if sr > 0 else 1
        merged = ensemble_continuous(g["starts"], g["preds"], length, kd)
        covered = np.zeros(length, dtype=bool)
        for s in g["starts"]:
            lo, hi = max(s, 0), min(s + K, length)
            if hi > lo:
                covered[lo:hi] = True
        merged = np.clip(_fill_uncovered(merged, covered), 0.0, 1.0)
        path = os.path.join(out_dir, sess, f"{role}.engagement.prediction.csv")
        write_csv_rows(path, [f"{float(v):.6f}" for v in merged])
    print(f"  wrote {len(groups)} files for {dataset}/{split}")


# ── categorical (kappa) jobs ──────────────────────────────────────────────────
@torch.no_grad()
def run_categorical(model, cfg, normalizer, device, split, out_dir, heads=("social", "task"),
                    inject=None):
    """Write ``{role}.{social,task}_engagement.prediction.csv`` for a PInSoRo test split.

    ``heads`` selects which engagement streams to write (``"social"``, ``"task"``, or both); only the
    requested heads are ensembled and emitted.

    By default reads the raw balance-trained logits and argmaxes — the head is calibrated at train
    time by the per-domain logit-adjustment margin (``pinsoro.train_logit_adjust_tau_{cc,cr}``).

    ``inject`` (optional) opts a head into an eval-time prior correction. It maps a head name to
    ``{"mode": "exact"|"beta", "beta": float, "target": π_target}``: ``"exact"`` fits a per-class
    bias so the realized argmax marginal *equals* ``π_target`` (``fit_marginal_bias``); ``"beta"``
    applies the one-shot ``beta·(log π_target − log π_pred)`` (``π_pred`` = the model's own argmax
    marginal).
    """
    inject = inject or {}
    heads = set(heads)
    want_s, want_t = "social" in heads, "task" in heads
    W, K = int(cfg["window"]["W"]), int(cfg["window"]["K"])
    kd = float(cfg.get("ensemble", {}).get("decay_kappa", 0.02))
    st = max(1, int(cfg.get("pinsoro", {}).get("pred_stride", 25)))
    workers = int(cfg["data"].get("eval_workers", cfg["data"].get("num_workers", 0)))
    _, loader = build_loader(cfg, ["pinsoro"], [split], train=False,
                             normalizer=normalizer, num_workers=workers)
    if loader is None:
        print(f"  no sessions for pinsoro/{split}")
        return

    feat_dirs: dict[str, str] = {}
    groups = defaultdict(lambda: {"starts": [], "sl": [], "tl": []})
    for batch in loader:
        out = model(move_to_device(batch, device))
        sl = out["social_logits"].float().cpu().numpy() if want_s else None   # (B, N, K, 5)
        tl = out["task_logits"].float().cpu().numpy() if want_t else None      # (B, N, K, 4)
        is_t = batch["is_target"].numpy()
        roles, sess, starts = batch["node_roles"], batch["session_id"], batch["window_start"]
        feat = batch["feature_dir"]
        B, N = is_t.shape
        for i in range(B):
            for nd in range(N):
                if is_t[i, nd] < 0.5:
                    continue
                key = (sess[i], roles[i][nd])
                feat_dirs[key] = feat[i]
                g = groups[key]
                g["starts"].append(int(starts[i]) + (W - K))
                if want_s:
                    g["sl"].append(subsample_forward_fill(sl[i, nd], st))
                if want_t:
                    g["tl"].append(subsample_forward_fill(tl[i, nd], st))

    # Ensemble every session first (so the social marginal can be measured before argmax).
    results = []
    for (sess, role), g in sorted(groups.items()):
        num, sr = native_length(feat_dirs[(sess, role)], role)            # native 30 fps grid
        t25 = max(int(round(num * registry.LABEL_FPS_CONTINUOUS / sr)), 1) if sr > 0 else 1
        covered = np.zeros(t25, dtype=bool)
        for s in g["starts"]:
            lo, hi = max(s, 0), min(s + K, t25)
            if hi > lo:
                covered[lo:hi] = True
        r = {"sess": sess, "role": role, "num": num, "covered": covered}
        if want_s:
            r["s_logit"] = ensemble_logits(g["starts"], g["sl"], t25, kd)     # (T25, 5)
        if want_t:
            r["t_logit"] = ensemble_logits(g["starts"], g["tl"], t25, kd)     # (T25, 4)
        results.append(r)

    # Prior-correct any requested head toward its target marginal (fit exactly, or one-shot β shift).
    domain = "cc" if split.endswith("cc") else "cr"
    for head, key in (("social", "s_logit"), ("task", "t_logit")):
        if head not in heads or head not in inject or not results:
            continue
        spec = inject[head]
        if spec["mode"] == "beta" and spec["beta"] == 0:
            continue
        _apply_prior_shift(results, key, spec, f"{domain}-{head}")

    for r in results:
        sess, role, num, covered = r["sess"], r["role"], r["num"], r["covered"]
        if want_s:
            s_idx = _fill_idx(r["s_logit"].argmax(-1), covered)
            s_native = resample_categorical_nearest(s_idx, num)               # 25 fps -> native 30 fps
            write_csv_rows(os.path.join(out_dir, sess, f"{role}.social_engagement.prediction.csv"),
                           [SOCIAL_BY_IDX[int(c)] for c in s_native])
        if want_t:
            t_idx = _fill_idx(r["t_logit"].argmax(-1), covered)
            t_native = resample_categorical_nearest(t_idx, num)
            write_csv_rows(os.path.join(out_dir, sess, f"{role}.task_engagement.prediction.csv"),
                           [TASK_BY_IDX[int(c)] for c in t_native])
    print(f"  wrote {len(results)} sessions for pinsoro/{split} ({'+'.join(sorted(heads))})")


def _fill_idx(idx: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Edge-hold/forward-fill class indices on uncovered frames (head/gaps/tail)."""
    cov = np.flatnonzero(covered)
    if cov.size == 0 or cov.size == idx.size:
        return idx
    nearest = np.interp(np.arange(idx.size), cov, cov).round().astype(np.int64)
    return idx[np.clip(nearest, 0, idx.size - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="comp_model/configs/default.yaml")
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint for all jobs (continuous CCC + PInSoRo kappa)")
    ap.add_argument("--out", default="submission",
                    help="output directory for the prediction tree")
    ap.add_argument("--zip", dest="zip_path", default="",
                    help="if set, also write a .zip of the submission tree at this path")
    ap.add_argument("--only", choices=["all", "continuous", "categorical"], default="all",
                    help="run only the continuous (CCC) or only the categorical (PInSoRo) jobs; "
                         "'categorical' regenerates just pinsoro-cc/pinsoro-cr in --out")
    ap.add_argument("--prior-inject", default="",
                    help="eval-time prior correction toward the local val-<domain> marginal, per "
                         "stream. Comma-separated stream=mode pairs; stream in "
                         "{cc-social,cc-task,cr-social,cr-task}, mode 'exact' (fit the argmax "
                         "marginal to equal val) or a float β (one-shot β·log(π_val/π_pred)). "
                         "E.g. 'cr-social=exact' or 'cr-social=exact,cc-task=0.3'. Streams not "
                         "listed write raw logits.")
    args = ap.parse_args()
    inject_specs = parse_prior_inject(args.prior_inject)   # {(domain, head): {"mode","beta"}}

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = cfg["data"]["norm_stats"]
    normalizer = (Normalizer.load(stats) if os.path.exists(stats) else empty_normalizer())

    model = EngagementModel(cfg, modality_dims=normalizer.effective_dims).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    print(f"loaded {args.checkpoint}")
    patch_language_embedding_to_mean(model, cfg)

    # Continuous jobs.
    if args.only in ("all", "continuous"):
        for folder, dataset, split in CONTINUOUS_JOBS:
            out_dir = os.path.join(args.out, folder)
            print(f"[{folder}] {dataset}/{split} (continuous)")
            run_continuous(model, cfg, normalizer, device, dataset, split, out_dir)

    # Categorical jobs: both heads (social + task) written per split.
    if args.only in ("all", "categorical"):
        # Resolve each requested injection's target prior once (read from local val-<domain> labels).
        targets = {dh: val_target_prior(cfg, *dh) for dh in inject_specs}
        for folder, split in CATEGORICAL_JOBS:
            out_dir = os.path.join(args.out, folder)
            print(f"[{folder}] pinsoro/{split} (categorical)")
            domain = "cc" if split.endswith("cc") else "cr"
            inject = {head: {**inject_specs[(domain, head)], "target": targets[(domain, head)]}
                      for head in ("social", "task") if (domain, head) in inject_specs}
            run_categorical(model, cfg, normalizer, device, split, out_dir,
                            heads=("social", "task"), inject=inject)

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(args.out):
                for fn in files:
                    fp = os.path.join(root, fn)
                    zf.write(fp, os.path.relpath(fp, args.out))
        print(f"zipped -> {args.zip_path}")


if __name__ == "__main__":
    main()
