#!/usr/bin/env python3
"""Compute window statistics for every PInSoRo (session, target child) and cache them in-session.

Writes ``<session_dir>/<role>.simple_feats.npz`` next to the raw streams, so the tables live with
the data they were derived from and the ``test-*`` splits (which have no annotations) are handled by
exactly the same path -- their label arrays are simply all ``-1``.

    python -m pinsoro_model.extract --splits train val test
    python -m pinsoro_model.extract --limit 2 --overwrite      # smoke test

Scored roles follow ``sessions.target_roles``: children only, so a child-child session yields two
tables and a child-robot session one (the robot is the partner node, never a target).
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import sessions

from . import columns as C
from . import features as F
from .windows import (CONTEXT_SEC, W_SEC, HOP_SEC, context_bounds, label_mode_purity,
                      run_lengths, window_bounds)

NPZ_SUFFIX = "simple_feats.npz"
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_schema.json")

# Smallest defensible feature set: is the child talking, is the partner talking, are they
# alternating, where is the child looking, and are their hands moving?
T0_FEATURES = [
    "room.eg.valid_frac.mean", "room.eg.active.frac", "room.eg.active_runs_per_min",
    "room.eg.active_run_mean_s", "room.eg.silence_run_max_s",
    "room.eg.activity_transition_rate", "room.eg.energy.mean", "room.eg.energy.std",
    "tgt.of2.gaze_angle_x.mean", "tgt.of2.gaze_angle_x.std", "tgt.of2.gaze_angle_x.sign_change_rate",
    "tgt.of2.gaze_angle_y.mean", "tgt.of2.gaze_angle_y.std", "tgt.of2.gaze_angle_y.sign_change_rate",
    "dy.abs_gaze_x_diff.mean", "dy.gaze_x_opposite_sign.frac",
    "tgt.of2.pose_Tz.mean", "tgt.of2.pose_Ry.mean", "tgt.of2.pose_Ry.std",
    "tgt.of2.head_trans_speed.mean",
    "tgt.op.RWrist.motion", "tgt.op.LWrist.motion",
    "tgt.op.Rwrist_height.mean", "tgt.op.Lwrist_height.mean",
    "tgt.of2.success.frac", "tgt.op.detect_rate",
    "dy.r.head_motion", "dy.r.hand_motion", "dy.r.AU12", "dy.r.gaze_angle_x",
    "dy.abs_Tz_diff.mean", "dy.both_face_tracked.frac",
    "dy.absdiff.op.hand_motion.mean", "dy.mean.op.hand_motion.mean",
    "tgt.of2.AU06_r.mean", "tgt.of2.AU12_r.mean", "tgt.of2.AU25_r.mean",
    "meta.domain_cc", "meta.age", "meta.gender", "meta.session_elapsed_frac",
]


def _tier_of(name: str) -> str:
    """T1 = per-child behaviour + dyadic + metadata; T2 adds env, full acoustics, long context."""
    if (name.startswith("env.") or re.search(r"\.egc\d\d\.", name)
            or name.startswith(("ctx30.", "ctx60.", "ctxd."))):
        return "T2"
    return "T1"


def session_row_features(tgt: F.NodeSeries, par: F.NodeSeries, env: F.NodeSeries | None,
                         feature_dir: str, role: str, domain: str, partner_is_robot: bool,
                         n_frames: int):
    """Feature dicts for every window of one (session, target role), in window order."""
    start, end, t_center = window_bounds(n_frames)
    duration_s = n_frames / C.VIDEO_FPS
    ctx = {sec: context_bounds(t_center, sec, n_frames) for sec in CONTEXT_SEC}
    rows = []
    for i in range(len(start)):
        lo, hi = int(start[i]), int(end[i])
        t_alo, t_ahi = tgt.audio_slice(lo, hi)
        row = {}
        row.update(F.block_a(tgt, lo, hi, "tgt"))
        row.update(F.block_b(tgt, lo, hi, "tgt"))
        row.update(F.block_c(tgt, t_alo, t_ahi, "room"))
        row.update(F.block_d(tgt, par, lo, hi))
        row.update(F.block_e(feature_dir, role, domain, partner_is_robot,
                             float(t_center[i]), duration_s))
        row.update(F.block_f(env, lo, hi))
        row.update(F.block_g(tgt, t_alo, t_ahi, "room"))
        # Block H: the same 24 scalars over 30 s / 60 s spans, plus the 60 s-vs-10 s drift.
        base = F.context_scalars(tgt, par, tgt, lo, hi, t_alo, t_ahi)
        wide = {}
        for sec in CONTEXT_SEC:
            cs, ce = ctx[sec]
            clo, chi = int(cs[i]), int(ce[i])
            ca_lo, ca_hi = tgt.audio_slice(clo, chi)
            wide[sec] = F.context_scalars(tgt, par, tgt, clo, chi, ca_lo, ca_hi)
            for k in F.H_SCALARS:
                row[f"ctx{int(sec)}.{k}"] = wide[sec][k]
        for k in F.H_SCALARS:
            row[f"ctxd.{k}"] = wide[CONTEXT_SEC[-1]][k] - base[k]
        rows.append(row)
    return rows, start, end, t_center


def read_frame_labels(path: str, class_map: dict, n_frames: int) -> np.ndarray:
    """Position-preserving read of a PInSoRo per-frame annotation CSV -> ``(n_frames,)`` int64.

    The file holds one line per video frame; an **empty line** or the literal ``nan`` marks an
    unannotated frame, and both conventions occur across the corpus. Do NOT use
    ``streams.read_str_csv_to_idx`` here: it drops blank lines, which compacts the array and
    silently shifts every label after the first gap. Coverage is far from complete -- some
    sessions are annotated over well under half their duration, and ``val-cr/014`` has no social
    annotation at all -- so the gaps are common, not an edge case.
    """
    lab = np.full(n_frames, -1, dtype=np.int64)
    if not path or not os.path.exists(path):
        return lab
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_frames:
                break
            s = line.strip().lower()
            if s:
                lab[i] = class_map.get(s, -1)
    return lab


def _labels(label_dir: str, role: str, axis: str, class_map: dict, n_frames: int):
    """Native-30 fps int label array for one axis, ``-1`` where absent (test splits have none)."""
    path = os.path.join(label_dir or "", f"{role}.{axis}_engagement.annotation.csv")
    return read_frame_labels(path if label_dir else "", class_map, n_frames)


def process_session(ref, overwrite: bool, verbose: bool = True):
    """Extract and write one session's ``.npz`` tables. Returns the number of files written."""
    domain = sessions.domain_of(ref.feature_dir)
    targets = sessions.target_roles(ref.roles, ref.feature_dir)
    if not targets:
        return 0
    child_nodes = [r for r in ref.roles if r in ("purple", "yellow")]
    out_paths = {r: os.path.join(ref.feature_dir, f"{r}.{NPZ_SUFFIX}") for r in targets}
    if not overwrite and all(os.path.exists(p) for p in out_paths.values()):
        if verbose:
            print(f"  skip {domain}/{ref.session_id} (exists)")
        return 0

    series = {r: F.NodeSeries(ref.feature_dir, r) for r in ref.roles}
    F.set_activity_threshold([series[r] for r in child_nodes])
    env = series.get("env")
    n_frames = min(series[r].T30 for r in ref.roles)

    written = 0
    for role in targets:
        partner = next((r for r in child_nodes if r != role), None)
        if partner is None:
            print(f"  !! {ref.session_id}/{role}: no partner node, skipped")
            continue
        tgt, par = series[role], series[partner]
        partner_is_robot = sessions.is_robot(ref.feature_dir, partner)

        rows, start, end, t_center = session_row_features(
            tgt, par, env, ref.feature_dir, role, domain, partner_is_robot, n_frames)
        names = list(rows[0].keys())
        X = np.empty((len(rows), len(names)), dtype=np.float32)
        for i, row in enumerate(rows):
            if list(row.keys()) != names:
                raise RuntimeError(f"feature key drift at window {i} of {ref.session_id}/{role}")
            X[i] = [row[k] for k in names]

        soc = _labels(ref.label_dir, role, "social", sessions.SOCIAL_CLASSES, n_frames)
        tsk = _labels(ref.label_dir, role, "task", sessions.TASK_CLASSES, n_frames)
        y_soc, p_soc = label_mode_purity(soc, start, end, sessions.NUM_SOCIAL_CLASSES)
        y_tsk, p_tsk = label_mode_purity(tsk, start, end, sessions.NUM_TASK_CLASSES)

        np.savez_compressed(
            out_paths[role],
            X=X, feat_names=np.array(names, dtype=np.str_),
            schema_hash=_hash(names),
            y_social=y_soc, y_task=y_tsk,
            purity_social=p_soc, purity_task=p_tsk,
            frame_start=start.astype(np.int32), frame_end=end.astype(np.int32),
            t_center=t_center, n_frames_30fps=np.int64(n_frames),
            session=ref.session_id, role=role, partner=partner, domain=domain, split=ref.split,
            partner_is_robot=bool(partner_is_robot),
            window_sec=np.float32(W_SEC), hop_sec=np.float32(HOP_SEC),
            social_runs=np.int64(len(run_lengths(soc)[1])),
            task_runs=np.int64(len(run_lengths(tsk)[1])),
        )
        written += 1
        if verbose:
            lab = (y_soc >= 0).mean()
            print(f"  {domain}/{ref.session_id}/{role}: {X.shape[0]} x {X.shape[1]}  "
                  f"labeled {lab:.0%}  nan {np.isnan(X).mean():.1%}")
    return written


def _hash(names) -> str:
    return hashlib.md5("\n".join(names).encode()).hexdigest()


def write_schema(names) -> None:
    """Persist the ordered feature names and their tier membership."""
    missing = [n for n in T0_FEATURES if n not in names]
    if missing:
        raise RuntimeError(f"T0 references features that are not emitted: {missing}")
    tiers = {n: _tier_of(n) for n in names}
    schema = {
        "n_features": len(names),
        "schema_hash": _hash(names),
        "window_sec": W_SEC, "hop_sec": HOP_SEC, "context_sec": list(CONTEXT_SEC),
        "counts": {
            "T0": len(T0_FEATURES),
            "T1": sum(1 for n in names if tiers[n] == "T1"),
            "T2": len(names),
        },
        "T0": T0_FEATURES,
        "tiers": tiers,
        "names": names,
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=1)
    print(f"schema -> {SCHEMA_PATH}  ({schema['counts']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/path/to/pinsoro")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="logical splits")
    ap.add_argument("--limit", type=int, default=None, help="process at most N sessions")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    refs = []
    for split in args.splits:
        refs.extend(sessions.iter_sessions(args.root, split))
    if args.limit:
        refs = refs[:args.limit]
    print(f"{len(refs)} session(s) over splits={args.splits}")

    t0 = time.time()
    total = 0
    for i, ref in enumerate(refs, 1):
        st = time.time()
        n = process_session(ref, args.overwrite)
        total += n
        if n:
            print(f"[{i}/{len(refs)}] {ref.split}/{ref.session_id}  {time.time() - st:.1f}s")

    # Refresh the schema from whatever was just written (any table has the full name list).
    for ref in refs:
        for role in sessions.target_roles(ref.roles, ref.feature_dir):
            p = os.path.join(ref.feature_dir, f"{role}.{NPZ_SUFFIX}")
            if os.path.exists(p):
                write_schema([str(x) for x in np.load(p, allow_pickle=False)["feat_names"]])
                print(f"wrote {total} table(s) in {time.time() - t0:.0f}s")
                return
    print(f"wrote {total} table(s) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
