#!/usr/bin/env python3
"""Precompute normalized 25 fps per-(session, role, modality) feature arrays for fast random access.

Materializes the normalized, channel-selected, 25 fps feature arrays once (fp16) so the loader can
memory-map and slice them instead of parsing/resampling/normalizing the raw streams on the hot path.

Output layout mirrors each session's feature dir relative to the dataset root::

    <precomputed_dir>/<dataset>/<relpath-of-feature_dir>/<role>.<modality>.npy   # (T, D_eff) float16

Only *present* modalities are written; an absent modality is simply a missing file (the loader fills
zeros, which the model masks out via ``node_present``). Both continuous (NoXi / NoXi-J / MPII) and
categorical (PInSoRo) datasets are precomputed: the loader memmap-slices whichever arrays exist and
falls back to the raw stream per missing/stale modality.

    python comp_model/scripts/precompute_features.py --config comp_model/configs/default.yaml \
        --splits train val
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from comp_model.config import load_config
from comp_model.data import registry
from comp_model.data.dataset import EngagementDataset
from comp_model.data.normalize import Normalizer, empty as empty_normalizer
from comp_model.data.sessions import build_index


def _session_out_dir(out_root: str, roots: dict, ref) -> str:
    """Mirror a session's feature_dir relative to its dataset root (matches the loader)."""
    rel = os.path.relpath(ref.feature_dir, roots[ref.dataset])
    return os.path.join(out_root, ref.dataset, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="comp_model/configs/default.yaml")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="datasets to precompute (default: all train_datasets, continuous + categorical)")
    ap.add_argument("--splits", nargs="*", default=["train", "val"],
                    help="logical splits to precompute (default: train val)")
    ap.add_argument("--precomputed_dir", default=None,
                    help="output root (default: data.precomputed_dir from the config)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-write arrays that already exist (default: skip existing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    roots = cfg["data"]["roots"]
    out_root = args.precomputed_dir or cfg["data"].get("precomputed_dir")
    if not out_root:
        sys.exit("no precomputed_dir: set data.precomputed_dir in the config or pass --precomputed_dir")

    stats_path = cfg["data"]["norm_stats"]
    normalizer = (Normalizer.load(stats_path) if os.path.exists(stats_path)
                  else empty_normalizer())
    if not os.path.exists(stats_path):
        print(f"[warn] {stats_path} missing — precomputing with identity normalization "
              f"(run fit_norm_stats.py first for the real stats).")

    datasets = args.datasets or cfg["data"]["train_datasets"]

    records = build_index(roots, datasets, args.splits)
    if not records:
        sys.exit(f"no records for datasets={datasets} splits={args.splits}")
    # EngagementDataset.__init__ computes per-record T (label-grid length) and reuses the exact
    # raw load+normalize path (_load_modality_raw).
    ds = EngagementDataset(records, cfg, normalizer=normalizer, train=False)

    print(f"precomputing {len(records)} sessions -> {out_root}  (datasets={datasets} splits={args.splits})")
    t0 = time.time()
    n_files = n_skip = 0
    n_bytes = 0
    for ri, rec in enumerate(ds.records):
        ref = rec.ref
        T = ds._T[ri]
        out_dir = _session_out_dir(out_root, roots, ref)
        os.makedirs(out_dir, exist_ok=True)
        wrote = []
        for role in rec.node_roles:
            for modality in registry.MODALITY_ORDER:
                outp = os.path.join(out_dir, f"{role}.{modality}.npy")
                if os.path.exists(outp) and not args.overwrite:
                    n_skip += 1
                    continue
                arr, present = ds._load_modality_raw(ref, role, modality, T)
                if not present:
                    continue                      # absent -> no file (loader fills zeros)
                arr = arr.astype(np.float16)
                np.save(outp, arr)
                n_files += 1
                n_bytes += arr.nbytes
                wrote.append(f"{role}.{modality}")
        print(f"  [{ri + 1:>3}/{len(records)}] {ref.dataset}/{ref.split}/{ref.session_id} "
              f"T={T}  +{len(wrote)} arrays", flush=True)

    print(f"\ndone: wrote {n_files} arrays ({n_bytes / 1e9:.2f} GB), skipped {n_skip} existing, "
          f"in {time.time() - t0:.0f}s")
    print(f"set `data.precomputed_dir: {out_root}` (already default if unchanged) and train.")


if __name__ == "__main__":
    main()
