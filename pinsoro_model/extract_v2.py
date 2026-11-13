#!/usr/bin/env python3
"""Compute the V2 feature set for every PInSoRo (session, target child).

    python -m pinsoro_model.extract_v2 --jobs 8
    python -m pinsoro_model.extract_v2 --limit 2      # smoke test

Writes ``<session_dir>/<role>.v2_feats.npz`` next to the raw streams. The existing
``<role>.simple_feats.npz`` files are never touched. Sessions are walked from PInSoRo's own on-disk
``{train,val,test}-{cc,cr}`` directories.

The window grid, session length rule and label reader are the same as ``extract.py``. Row *i* of a
V2 table therefore describes the same 10 s as row *i* of the matching ``simple_feats`` table, which
is what lets the re-extraction be spot-checked column for column.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import sessions
from .streams import read_stream_header

from . import columns as C
from . import features as F
from .data import PINSORO_ROOT, iter_split_sessions
from .extract import _labels
from .feature_sets import CONTEXT_CORE, ROLE_SIGN, V2_ALL, V2_SUFFIX
from .windows import HOP_SEC, W_SEC, context_bounds, label_mode_purity, run_lengths, window_bounds

CTX_SEC = 60.0
AU = C.AU_R_INDEX


def per_child(s: F.NodeSeries, lo: int, hi: int, sign: float) -> dict:
    """The 25 per-child features over frames ``[lo, hi)``."""
    gx, gy = s.gaze_x[lo:hi], s.gaze_y[lo:hi]
    yaw, tz = s.pose_r[lo:hi, 1], s.pose_t[lo:hi, 2]
    au = s.au_r[lo:hi]
    a12, a25 = au[:, AU["AU12"]], au[:, AU["AU25"]]
    nk = C.KP["neck"]
    f = {
        "gaze_x_aligned.mean": sign * F.nmean(gx),
        "gaze_x.std": F.nstd(gx),
        "gaze_y.mean": F.nmean(gy),
        "gaze_y.std": F.nstd(gy),
        "gaze_shift_rate": F.sign_change_rate(gx),
        "head_yaw_aligned.mean": sign * F.nmean(yaw),
        "head_yaw.std": F.nstd(yaw),
        "head_pitch.mean": F.nmean(s.pose_r[lo:hi, 0]),
        "head_dist.mean": F.nmean(tz),
        "head_dist.slope": F.slope(tz),
        "head_rot_speed.mean": F.nmean(s.head_rot_speed[lo:hi]),
        "hand_motion.mean": F.nmean(s.hand_motion[lo:hi]),
        "RWrist.motion": F.nmean(s.kp_speed[lo:hi, C.KP["RWrist"]]),
        "LWrist.motion": F.nmean(s.kp_speed[lo:hi, C.KP["LWrist"]]),
        # image y grows downward, so height above the neck is the negated offset.
        "Rwrist_height.mean": F.nmean(-(s.yn[lo:hi, C.KP["RWrist"]] - s.yn[lo:hi, nk])),
        "Lwrist_height.mean": F.nmean(-(s.yn[lo:hi, C.KP["LWrist"]] - s.yn[lo:hi, nk])),
        "nose_height.mean": -F.nmean(s.yn[lo:hi, C.KP["nose"]]),
        "AU06.mean": F.nmean(au[:, AU["AU06"]]),
        "AU12.mean": F.nmean(a12),
        "AU12.frac_smiling": (float(np.nanmean(a12 > 1.0))
                              if F._n(a12) >= F.MIN_VALID else np.nan),
        "AU04.mean": F.nmean(au[:, AU["AU04"]]),
        "AU25.mean": F.nmean(a25),
        "AU25.std": F.nstd(a25),
        "face_tracked.frac": F.nfrac(s.of_ok[lo:hi]),
        "body_detected.frac": F.nfrac(s.kp_valid[lo:hi]),
    }
    return f


def dyadic(t: F.NodeSeries, p: F.NodeSeries, lo: int, hi: int, psign: float) -> dict:
    """The 8 target-vs-partner features. Computed for cr too, but excluded there at training."""
    return {
        "r.head_motion": F.corr(t.head_trans_speed[lo:hi], p.head_trans_speed[lo:hi]),
        "r.hand_motion": F.corr(t.hand_motion[lo:hi], p.hand_motion[lo:hi]),
        "r.smile": F.corr(t.au_r[lo:hi, AU["AU12"]], p.au_r[lo:hi, AU["AU12"]]),
        "r.mouth": F.corr(t.au_r[lo:hi, AU["AU25"]], p.au_r[lo:hi, AU["AU25"]]),
        "r.gaze_x": F.corr(t.gaze_x[lo:hi], p.gaze_x[lo:hi]),
        "partner.hand_motion.mean": F.nmean(p.hand_motion[lo:hi]),
        "partner.smile.mean": F.nmean(p.au_r[lo:hi, AU["AU12"]]),
        "partner.gaze_x_aligned.mean": psign * F.nmean(p.gaze_x[lo:hi]),
    }


def core(t: F.NodeSeries, p: F.NodeSeries, lo: int, hi: int, sign: float) -> dict:
    """The context signals of ``CONTEXT_CORE`` over an arbitrary frame span."""
    out = {
        "gaze_x_aligned.mean": sign * F.nmean(t.gaze_x[lo:hi]),
        "gaze_y.mean": F.nmean(t.gaze_y[lo:hi]),
        "head_yaw_aligned.mean": sign * F.nmean(t.pose_r[lo:hi, 1]),
        "head_dist.mean": F.nmean(t.pose_t[lo:hi, 2]),
        "hand_motion.mean": F.nmean(t.hand_motion[lo:hi]),
        "AU12.mean": F.nmean(t.au_r[lo:hi, AU["AU12"]]),
        "AU25.std": F.nstd(t.au_r[lo:hi, AU["AU25"]]),
        "r.hand_motion": F.corr(t.hand_motion[lo:hi], p.hand_motion[lo:hi]),
    }
    assert list(out) == CONTEXT_CORE
    return out


def _session_frames(sdir: str, roles, series: dict) -> int:
    """Session length on the 30 fps grid -- the same rule as ``extract.py``: the shortest of every
    node's OpenFace/OpenPose streams, env included. The env length is read from stream headers only."""
    lens = [series[r].T30 for r in series]
    if "env" in roles:
        for mod in ("openface2", "openpose"):
            path = os.path.join(sdir, f"env.{mod}.stream")
            if os.path.exists(path):
                lens.append(read_stream_header(path)[0])
    return int(min(lens))


def _hash(names) -> str:
    return hashlib.md5("\n".join(names).encode()).hexdigest()


def process_session(job) -> list:
    split, domain, sid, sdir, overwrite = job
    roles = sessions.detect_roles(sdir)
    targets = sessions.target_roles(roles, sdir)
    children = [r for r in roles if r in ("purple", "yellow")]
    out_paths = {r: os.path.join(sdir, f"{r}.{V2_SUFFIX}") for r in targets}
    if not targets:
        return [f"  skip {split}-{domain}/{sid}: no target roles"]
    if not overwrite and all(os.path.exists(p) for p in out_paths.values()):
        return [f"  skip {split}-{domain}/{sid}: exists"]

    series = {r: F.NodeSeries(sdir, r) for r in children}
    n_frames = _session_frames(sdir, roles, series)
    start, end, t_center = window_bounds(n_frames)
    cs, ce = context_bounds(t_center, CTX_SEC, n_frames)

    msgs = []
    for role in targets:
        partner = next(r for r in children if r != role)
        t, p = series[role], series[partner]
        sign, psign = ROLE_SIGN[role], ROLE_SIGN[partner]
        X = np.empty((len(start), len(V2_ALL)), np.float32)
        for i in range(len(start)):
            lo, hi = int(start[i]), int(end[i])
            row = per_child(t, lo, hi, sign)
            row.update(dyadic(t, p, lo, hi, psign))
            base = core(t, p, lo, hi, sign)
            wide = core(t, p, int(cs[i]), int(ce[i]), sign)
            for k in CONTEXT_CORE:
                row[f"ctx60.{k}"] = wide[k]
                row[f"ctxd.{k}"] = wide[k] - base[k]
            if i == 0 and set(row) != set(V2_ALL):
                raise RuntimeError(f"V2 key mismatch: {sorted(set(row) ^ set(V2_ALL))}")
            X[i] = [row[n] for n in V2_ALL]

        soc = _labels(sdir, role, "social", sessions.SOCIAL_CLASSES, n_frames)
        tsk = _labels(sdir, role, "task", sessions.TASK_CLASSES, n_frames)
        y_soc, p_soc = label_mode_purity(soc, start, end, sessions.NUM_SOCIAL_CLASSES)
        y_tsk, p_tsk = label_mode_purity(tsk, start, end, sessions.NUM_TASK_CLASSES)
        np.savez_compressed(
            out_paths[role],
            X=X, feat_names=np.array(V2_ALL, dtype=np.str_), schema_hash=_hash(V2_ALL),
            y_social=y_soc, y_task=y_tsk, purity_social=p_soc, purity_task=p_tsk,
            frame_start=start.astype(np.int32), frame_end=end.astype(np.int32),
            t_center=t_center, n_frames_30fps=np.int64(n_frames),
            session=sid, role=role, partner=partner, domain=domain, split=split,
            partner_is_robot=bool(sessions.is_robot(sdir, partner)),
            role_sign=np.float32(sign), window_sec=np.float32(W_SEC), hop_sec=np.float32(HOP_SEC),
            social_runs=np.int64(len(run_lengths(soc)[1])),
            task_runs=np.int64(len(run_lengths(tsk)[1])),
        )
        msgs.append(f"  {split}-{domain}/{sid}/{role}: {X.shape[0]} x {X.shape[1]}  "
                    f"labeled {(y_soc >= 0).mean():.0%}  nan {np.isnan(X).mean():.1%}")
    return msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=PINSORO_ROOT)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--domains", nargs="*", default=["cc", "cr"])
    ap.add_argument("--limit", type=int, default=None, help="process at most N sessions")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = [(split, dom, sid, sdir, args.overwrite)
            for split in args.splits for dom in args.domains
            for sid, sdir in iter_split_sessions(args.root, split, dom)]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"V2 extraction: {len(jobs)} sessions, {len(V2_ALL)} features, {args.jobs} workers",
          flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_session, j): j for j in jobs}
        for fut in as_completed(futs):
            done += 1
            for m in fut.result():
                print(m, flush=True)
            print(f"[{done}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
