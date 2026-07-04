#!/usr/bin/env python3
"""Fit per-modality normalization stats on the TRAIN split only.

Accumulates mean/std per modality from raw (un-normalized) streams over the train
sessions of the configured train datasets, then writes an .npz consumed by
data.normalize.Normalizer. Never touches val/test.

Usage:
    python comp_model/scripts/fit_norm_stats.py \
        --config comp_model/configs/default.yaml \
        [--max-sessions-per-dataset N] [--frame-stride S] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from comp_model.config import load_config
from comp_model.data import registry, streams
from comp_model.data.class_weights import pinsoro_class_counts
from comp_model.data.sessions import build_index


def _iter_target_labels(rec):
    """Yield the raw continuous engagement label vector for each *scored* role.

    Only meaningful for continuous datasets (NoXi / NoXi-J); categorical sessions have no
    continuous annotation and are skipped by the caller. Non-finite entries (``-nan(ind)`` in
    a few annotation files) are dropped so they never bias the standardization stats.
    """
    ref = rec.ref
    if ref.label_kind != "continuous" or not ref.label_dir:
        return
    for role in rec.target_roles:
        path = os.path.join(ref.label_dir, f"{role}.engagement.annotation.csv")
        if not os.path.exists(path):
            continue
        v = np.genfromtxt(path, dtype=np.float64)
        v = np.atleast_1d(v)
        v = v[np.isfinite(v)]
        if v.size:
            yield v


def _iter_node_modalities(rec):
    """Yield (modality, raw_array[T_native, D]) over *all* nodes in the session, raw values."""
    ref = rec.ref
    for role in rec.node_roles:
        for modality in registry.MODALITY_ORDER:
            pattern, kind = registry.MODALITY_FILE[modality]
            fname = pattern.format(role=role)
            if modality == "whisper":
                if not ref.whisper_dir:
                    continue
                path = os.path.join(ref.whisper_dir, fname)
                if not os.path.exists(path):
                    continue
                arr = np.load(path).astype(np.float32)
            else:
                path = os.path.join(ref.feature_dir, fname)
                if not streams.file_exists(path):
                    continue
                arr, _ = streams.read_stream(path)
            if arr.ndim == 2 and arr.shape[1] == registry.MODALITY_DIMS[modality]:
                yield modality, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="comp_model/configs/default.yaml")
    ap.add_argument("--max-sessions-per-dataset", type=int, default=0,
                    help="0 = all train sessions")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--robust-stride", type=int, default=8,
                    help="extra subsample stride for robust-modality quantile estimation "
                         "(on top of --frame-stride); keeps the value buffer bounded")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    roots = cfg["data"]["roots"]
    datasets = cfg["data"]["train_datasets"]
    # Fit over the same splits the trainer pools in (default train+val) so MPII — labelled only on
    # its `val` split — contributes normalization stats.
    train_splits = list(cfg["data"].get("train_splits", ["train"]))
    out_path = args.out or cfg["data"]["norm_stats"]

    # Running sums in float64 per modality, with per-column NaN-aware counts.
    dims = registry.MODALITY_DIMS
    s1 = {m: np.zeros(dims[m], np.float64) for m in registry.MODALITY_ORDER}
    s2 = {m: np.zeros(dims[m], np.float64) for m in registry.MODALITY_ORDER}
    cnt = {m: np.zeros(dims[m], np.float64) for m in registry.MODALITY_ORDER}
    # Scalar running sums for the continuous engagement target (diffusion standardization).
    # Pooled across datasets (fallback) plus per-dataset sums: the target is standardized by its own
    # dataset's (mean, std).
    ly1 = ly2 = lyn = 0.0
    ldy1: dict = {}
    ldy2: dict = {}
    ldyn: dict = {}

    # Robust modalities (median/IQR scaling) need raw values, not just moments. Collect a strided
    # subsample per channel (with NaN at invalid frames); the final percentile pass is NaN-aware.
    robust_mods = {m for m in registry.MODALITY_ORDER
                   if registry.MODALITY_NORM_MODE.get(m) == "robust"}
    rob_buf: dict[str, list] = {m: [] for m in robust_mods}
    ROBUST_SAMPLE_STRIDE = int(args.robust_stride)

    # Per-session modalities (instance centering) scale by the within-session std. Accumulate the
    # pooled within-stream sum of squared deviations from each node-stream's own mean.
    persess_mods = {m for m in registry.MODALITY_ORDER
                    if registry.MODALITY_NORM_MODE.get(m) == "per_session"}
    ws_ss = {m: np.zeros(dims[m], np.float64) for m in persess_mods}
    ws_n = {m: np.zeros(dims[m], np.float64) for m in persess_mods}

    for dataset in datasets:
        records = build_index(roots, [dataset], train_splits)
        if args.max_sessions_per_dataset:
            records = records[: args.max_sessions_per_dataset]
        print(f"[{dataset}] {len(records)} session records (splits={train_splits})")
        for ri, rec in enumerate(records):
            for modality, arr in _iter_node_modalities(rec):
                a = arr[:: args.frame_stride].astype(np.float64)
                # finite + in-range, plus per-modality sentinel floor (e.g. OpenPose -1).
                valid = streams.valid_feature_mask(
                    a, valid_min=registry.MODALITY_VALID_MIN.get(modality))
                if modality in robust_mods:
                    rows = a[::ROBUST_SAMPLE_STRIDE]
                    vmask = valid[::ROBUST_SAMPLE_STRIDE]
                    rob_buf[modality].append(np.where(vmask, rows, np.nan))
                av = np.where(valid, a, 0.0)
                if modality in persess_mods:
                    # within-stream deviations from this node-stream's own per-channel mean.
                    sn = valid.sum(axis=0)
                    smean = av.sum(axis=0) / np.maximum(sn, 1.0)
                    dev = np.where(valid, a - smean, 0.0)
                    ws_ss[modality] += (dev * dev).sum(axis=0)
                    ws_n[modality] += sn
                s1[modality] += av.sum(axis=0)
                s2[modality] += (av * av).sum(axis=0)
                cnt[modality] += valid.sum(axis=0)
            for lab in _iter_target_labels(rec):
                s1l = float(lab.sum())
                s2l = float((lab * lab).sum())
                nl = int(lab.size)
                ly1 += s1l; ly2 += s2l; lyn += nl
                ldy1[dataset] = ldy1.get(dataset, 0.0) + s1l
                ldy2[dataset] = ldy2.get(dataset, 0.0) + s2l
                ldyn[dataset] = ldyn.get(dataset, 0) + nl
            if (ri + 1) % 10 == 0:
                print(f"  ...{ri + 1}/{len(records)}")

    # Robust center/scale (median, NIQR) for robust modalities — NaN-aware over the subsample so
    # masked frames are ignored.  Computed once here; consumed per modality below.
    robust_stats = {}
    for m in robust_mods:
        if not rob_buf[m]:
            continue
        vals = np.concatenate(rob_buf[m], axis=0)          # (S, D_raw), NaN at invalid frames
        med = np.nanmedian(vals, axis=0)
        p25, p75 = np.nanpercentile(vals, [25, 75], axis=0)
        niqr = (p75 - p25) / 1.349                          # normal-consistent IQR scale
        robust_stats[m] = (med, niqr, int(vals.shape[0]))
        rob_buf[m] = None                                   # free the buffer

    out = {}
    for m in registry.MODALITY_ORDER:
        n = np.maximum(cnt[m], 1.0)
        if cnt[m].sum() == 0:
            print(f"  WARN: no data for modality {m}; skipping")
            continue
        mean = s1[m] / n
        var = np.maximum(s2[m] / n - mean * mean, 0.0)
        std = np.sqrt(var)

        # Build keep_channels: all channels except structural drops and data-dead (std=0).
        structural_drop = set(registry.MODALITY_DROP_CHANNELS.get(m, []))
        data_dead = set(int(i) for i in np.where(std == 0)[0])
        all_drop = structural_drop | data_dead
        keep = np.array([c for c in range(registry.MODALITY_DIMS[m]) if c not in all_drop],
                        dtype=np.int64)
        n_dropped = len(all_drop)
        if n_dropped:
            struct_d = sorted(structural_drop)
            dead_d   = sorted(data_dead - structural_drop)
            print(f"  {m}: dropping {n_dropped} channels "
                  f"(structural={struct_d}, data-dead={dead_d[:8]}{'…' if len(dead_d)>8 else ''})")

        mean_kept = mean[keep].astype(np.float32)
        var_kept = var[keep]

        norm_mode = registry.MODALITY_NORM_MODE.get(m, "z_score")
        if norm_mode == "global_std":
            # Subtract per-dim mean but divide by a single global scalar. Store it replicated to
            # (D_eff,) so apply() is unchanged.
            global_std = float(np.sqrt(np.mean(var_kept))) if var_kept.size > 0 else 1.0
            global_std = max(global_std, 1e-6)
            std_kept = np.full(len(keep), global_std, dtype=np.float32)
            print(f"  {m}: D_raw={registry.MODALITY_DIMS[m]} -> D_eff={len(keep)}  "
                  f"n_frames~{int(cnt[m].max())}  mode=global_std  "
                  f"global_std={global_std:.4f}  "
                  f"per-dim std range=[{std[keep].min():.4f}, {std[keep].max():.4f}]")
        elif norm_mode == "robust" and m in robust_stats:
            # Center on the median and scale by NIQR.
            med, niqr, n_samp = robust_stats[m]
            floor = 1e-6
            center = np.where(np.isfinite(med[keep]), med[keep], mean[keep])
            # Near-zero IQR (constant-bulk channels) -> fall back to the moment std, never divide ~0.
            niqr_k = niqr[keep]
            scale = np.where(niqr_k > floor, niqr_k, np.maximum(std[keep], floor))
            mean_kept = center.astype(np.float32)
            std_kept = scale.astype(np.float32)
            n_fb = int(np.sum(~(niqr_k > floor)))
            ratio = std[keep] / np.maximum(std_kept, floor)
            print(f"  {m}: D_raw={registry.MODALITY_DIMS[m]} -> D_eff={len(keep)}  "
                  f"n_frames~{int(cnt[m].max())}  mode=robust  "
                  f"quantile_samples={n_samp}  iqr_fallback={n_fb}  "
                  f"max std/robust_scale={float(ratio.max()):.1f}")
        elif norm_mode == "per_session" and m in ws_n:
            # Scale by within-session std; the center is each node-stream's own mean at load time.
            # Stored mean is the fallback center (used only for all-invalid channels in a stream).
            floor = 1e-6
            within = np.sqrt(ws_ss[m] / np.maximum(ws_n[m], 1.0))
            within_k = within[keep]
            std_kept = np.where(within_k > floor, within_k,
                                np.maximum(std[keep], floor)).astype(np.float32)
            n_fb = int(np.sum(~(within_k > floor)))
            compress = np.nanmedian(std[keep] / np.maximum(std_kept, floor))
            print(f"  {m}: D_raw={registry.MODALITY_DIMS[m]} -> D_eff={len(keep)}  "
                  f"n_frames~{int(cnt[m].max())}  mode=per_session  "
                  f"within_floor_fallback={n_fb}  median global/within = {float(compress):.2f}")
        else:
            std_kept = std[keep].astype(np.float32)
            print(f"  {m}: D_raw={registry.MODALITY_DIMS[m]} -> D_eff={len(keep)}  "
                  f"n_frames~{int(cnt[m].max())}  "
                  f"mean[0]={mean_kept[0]:.4f}  std[0]={std_kept[0]:.4f}")

        out[f"{m}.keep_channels"] = keep
        out[f"{m}.mean"] = mean_kept
        out[f"{m}.std"] = std_kept

    if lyn > 0:
        lmean = ly1 / lyn
        lstd = np.sqrt(max(ly2 / lyn - lmean * lmean, 0.0))
        out["label_continuous.mean"] = np.float32([lmean])
        out["label_continuous.std"] = np.float32([lstd])
        print(f"  label_continuous (pooled): n_frames~{int(lyn)} mean={lmean:.4f} std={lstd:.4f}")
        # Per-dataset stats: each continuous dataset's target is standardized by its own (mean, std).
        for ds in sorted(ldyn):
            if ldyn[ds] <= 0:
                continue
            dm = ldy1[ds] / ldyn[ds]
            dstd = np.sqrt(max(ldy2[ds] / ldyn[ds] - dm * dm, 0.0))
            out[f"label_continuous.{ds}.mean"] = np.float32([dm])
            out[f"label_continuous.{ds}.std"] = np.float32([dstd])
            print(f"  label_continuous ({ds}): n_frames~{int(ldyn[ds])} "
                  f"mean={dm:.4f} std={dstd:.4f}")
    else:
        print("  WARN: no continuous labels found; diffusion target standardization disabled")

    # PInSoRo class counts (train-only), stored as a diagnostic of the natural class imbalance.
    soc_cnt, tsk_cnt = pinsoro_class_counts(roots, datasets, ["train"])
    if soc_cnt.sum() or tsk_cnt.sum():
        out["pinsoro_social_counts"] = soc_cnt.astype(np.int64)
        out["pinsoro_task_counts"] = tsk_cnt.astype(np.int64)
        print(f"  pinsoro class counts: social={soc_cnt.tolist()} task={tsk_cnt.tolist()}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **out)
    n_mod = sum(1 for k in out if k.endswith(".mean") and not k.startswith("label_"))
    print(f"Wrote {out_path} ({n_mod} modalities + label_continuous)")


if __name__ == "__main__":
    main()
