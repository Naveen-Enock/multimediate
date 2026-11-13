#!/usr/bin/env python3
"""Write a MultiMediate'26 challenge submission from a trained checkpoint.

Runs test-set inference and writes per-frame prediction CSVs in the official folder structure, then
(optionally) zips them. The submission tree is::

    <out>/
      noxi-base/<sess>/{expert,novice}.engagement.prediction.csv          (CCC, 25 fps, [0,1])
      noxi-additional/<sess>/{expert,novice}.engagement.prediction.csv
      noxi-j/<sess>/{expert,novice}.engagement.prediction.csv
      mpiigroupinteraction/<sess>/subjectPos{1..4}.engagement.prediction.csv

PInSoRo (kappa) predictions come from ``pinsoro_model``.

Inference mirrors ``eval/validate.py`` but writes **absolute-frame** sequences over the *full*
session: each window predicts its centered ``K``-frame chunk ``[start+(W-K)//2, start+(W-K)//2+K)``
(leading/trailing context of ``(W-K)//2`` frames each side; with the default ``S == K`` consecutive
chunks are back-to-back, not overlapping), so chunks are placed at their true absolute start
``start+(W-K)//2`` and both head frames ``[0, (W-K)//2)`` and tail frames past the last chunk (plus
any interior gaps) are filled by interpolation/edge-hold over the covered frames.
Sessions write the 25 fps grid directly. Predictions are written raw by default;
``--continuous-calibrate`` opts a dataset into an eval-time affine recalibration toward its local
val-split label ``(mean, std)`` (``val_target_moments`` / ``_calibrate_continuous``), for a dataset
whose label scale/location drifts between train and val (e.g. NoXi-J).

Language embedding patch (always on): NoXi-additional sessions use languages absent from train+val,
whose ``language_emb`` rows never received a training gradient. Before inference we overwrite every
unseen language row with the mean of the seen rows; a no-op for every split except noxi-additional.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from continuous_model.config import load_config
from continuous_model.data import registry, sessions, streams
from continuous_model.data.build import build_loader
from continuous_model.data.normalize import Normalizer, empty as empty_normalizer
from continuous_model.models.engagement_model import EngagementModel
from continuous_model.training.ensemble import ensemble_continuous
from continuous_model.training.trainer import move_to_device

# Continuous (CCC) jobs: (output folder, dataset key, logical split).
CONTINUOUS_JOBS = [
    ("noxi-base",            "noxi",    "test_base"),
    ("noxi-additional",      "noxi",    "test_additional"),
    ("noxi-j",               "noxi_j",  "test"),
    ("mpiigroupinteraction", "mpii",    "test"),
]


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


# ── continuous prior correction (per dataset) ─────────────────────────────────
def val_target_moments(cfg, dataset: str) -> tuple[float, float]:
    """Empirical ``(mean, std)`` of ``{dataset}``'s continuous engagement label on its local val split.

    The best available estimate of the test-split target distribution for a dataset whose label
    scale/location shifts between train and val (e.g. NoXi-J, whose mean/std both drift from train to
    val). ``run_continuous``'s ``calibrate`` arg recalibrates that dataset's ensembled predictions
    toward this ``(mean, std)`` via moment matching.
    """
    records = sessions.build_index(cfg["data"]["roots"], [dataset], ["val"])
    vals = []
    for rec in records:
        for role in rec.target_roles:
            fp = os.path.join(rec.ref.label_dir, f"{role}.engagement.annotation.csv")
            if not os.path.exists(fp):
                continue
            v = np.genfromtxt(fp, dtype=np.float64)
            v = np.atleast_1d(v)
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v)
    if not vals:
        raise RuntimeError(f"no val annotations found for {dataset} (val_target_moments)")
    allv = np.concatenate(vals)
    mean, std = float(allv.mean()), float(allv.std())
    print(f"  {dataset} target moments (val): mean={mean:.4f} std={std:.4f}")
    return mean, std


def _calibrate_continuous(results: list, target: tuple[float, float], tag: str) -> None:
    """Recalibrate every ``results[i]["values"]`` in place toward ``target = (mean, std)``.

    Fits the affine correction ``y' = a·y + b`` on the job's own pooled realized ``(mean, std)``
    vs. ``target``, with ``a = σ_target/σ_pred``, ``b = μ_target − a·μ_pred``. This is the CCC-optimal
    recalibration for a fixed prediction *shape*: for any correlation ρ between predictions and the
    true target, this (a, b) maximizes ``CCC = 2·a·ρ·σ_pred·σ_target / (a²σ_pred² + σ_target² +
    (a·μ_pred+b−μ_target)²)`` and it collapses to exactly ``CCC = ρ``. It removes bias/scale
    mismatch only — it cannot manufacture correlation that isn't already in the predictions.
    """
    target_mean, target_std = target
    pooled = np.concatenate([r["values"] for r in results])
    pred_mean, pred_std = float(pooled.mean()), float(pooled.std())
    a = target_std / max(pred_std, 1e-8)
    b = target_mean - a * pred_mean
    for r in results:
        r["values"] = a * r["values"] + b
    print(f"  {tag} calibrated: pred(mean={pred_mean:.4f}, std={pred_std:.4f}) "
          f"-> target(mean={target_mean:.4f}, std={target_std:.4f})  a={a:.4f} b={b:+.4f}")


# ── per-frame reconstruction helpers ──────────────────────────────────────────
def _fill_uncovered(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Fill frames with no window coverage by interpolation + edge-hold over covered frames.

    ``values``/``covered``: ``(T,)`` and ``(T,)`` bool. With centered chunks, both head frames
    ``[0, (W-K)//2)`` (before the earliest chunk) and tail frames (after the last chunk) go
    uncovered — ``np.interp`` edge-holds at both ends (constant extrapolation outside the covered
    range) and interpolates any interior gaps.
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
def run_continuous(model, cfg, normalizer, device, dataset, split, out_dir, calibrate=None):
    """Write ``{role}.engagement.prediction.csv`` for every scored node of a continuous test split.

    ``calibrate`` (optional) is a ``(mean, std)`` target — typically ``val_target_moments(cfg,
    dataset)`` — that recalibrates every ensembled prediction in this job via
    ``_calibrate_continuous`` before it's clipped to ``[0, 1]`` and written.
    """
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
        pred = normalizer.destandardize_label(dataset, pred)   # standardized-scale -> raw [0, 1]-ish
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
                g["starts"].append(int(starts[i]) + (W - K) // 2)   # true absolute chunk start
                g["preds"].append(pred[i, nd])

    results = []
    for (sess, role), g in sorted(groups.items()):
        num, sr = native_length(feat_dirs[(sess, role)], role)
        length = max(int(round(num * registry.LABEL_FPS_CONTINUOUS / sr)), 1) if sr > 0 else 1
        merged = ensemble_continuous(g["starts"], g["preds"], length, kd)
        covered = np.zeros(length, dtype=bool)
        for s in g["starts"]:
            lo, hi = max(s, 0), min(s + K, length)
            if hi > lo:
                covered[lo:hi] = True
        values = _fill_uncovered(merged, covered)
        results.append({"sess": sess, "role": role, "values": values})

    if calibrate is not None and results:
        _calibrate_continuous(results, calibrate, f"{dataset}/{split}")

    for r in results:
        values = np.clip(r["values"], 0.0, 1.0)
        path = os.path.join(out_dir, r["sess"], f"{r['role']}.engagement.prediction.csv")
        write_csv_rows(path, [f"{float(v):.6f}" for v in values])
    print(f"  wrote {len(results)} files for {dataset}/{split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="continuous_model/configs/default.yaml")
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint to run")
    ap.add_argument("--eval-stride", type=int, default=None,
                    help="override window.eval_S: slide inference windows this many frames so "
                         "chunks overlap and training/ensemble.py merges K/stride predictions per "
                         "frame (must be < window.K to have any effect)")
    ap.add_argument("--out", default="submission",
                    help="output directory for the prediction tree")
    ap.add_argument("--zip", dest="zip_path", default="",
                    help="if set, also write a .zip of the submission tree at this path")
    ap.add_argument("--continuous-calibrate", default="",
                    help="eval-time affine recalibration (moment matching) of continuous "
                         "predictions toward each listed dataset's local val-split label "
                         "(mean, std) — comma-separated dataset keys, e.g. 'noxi_j' or "
                         "'noxi,noxi_j'. Datasets not listed write raw predictions.")
    args = ap.parse_args()
    calibrate_datasets = {d.strip() for d in args.continuous_calibrate.split(",") if d.strip()}

    cfg = load_config(args.config)
    if args.eval_stride is not None:
        cfg["window"]["eval_S"] = int(args.eval_stride)
        print(f"  eval stride {args.eval_stride} (K={cfg['window']['K']}) — temporal ensembling "
              f"merges ~{max(1, cfg['window']['K'] // args.eval_stride)} chunks/frame")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = cfg["data"]["norm_stats"]
    normalizer = (Normalizer.load(stats) if os.path.exists(stats) else empty_normalizer())

    model = EngagementModel(cfg, modality_dims=normalizer.effective_dims).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    print(f"loaded {args.checkpoint}")
    patch_language_embedding_to_mean(model, cfg)

    calib_cache: dict = {}
    for folder, dataset, split in CONTINUOUS_JOBS:
        out_dir = os.path.join(args.out, folder)
        print(f"[{folder}] {dataset}/{split}")
        calib = None
        if dataset in calibrate_datasets:
            calib = calib_cache.setdefault(dataset, val_target_moments(cfg, dataset))
        run_continuous(model, cfg, normalizer, device, dataset, split, out_dir, calibrate=calib)

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(args.out):
                for fn in files:
                    fp = os.path.join(root, fn)
                    zf.write(fp, os.path.relpath(fp, args.out))
        print(f"zipped -> {args.zip_path}")


if __name__ == "__main__":
    main()
