#!/usr/bin/env python3
"""Run a fitted run's models over a split and write challenge-format predictions.

    python -m pinsoro_model.predict --run-name simple01 --tier T1 --splits test
    python -m pinsoro_model.predict --run-name simple01 --tier T1 --splits val --score

Loads ``runs/<run>/models/<domain>_<axis>_<tier>.json`` together with the decode state saved next
to it (transition matrix, class prior, and the stickiness / prior-division that CV selected), so
inference reproduces exactly the decoding the reported kappa was measured with.

Output mirrors the challenge layout::

    runs/<run>/submission/pinsoro-<domain>/<session>/<role>.<axis>_engagement.prediction.csv

one class string per 30 fps frame. ``--score`` additionally reports per-frame kappa, which is only
meaningful for annotated splits (``test-*`` ships no annotations).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .kappa import cohen_kappa

from . import data as D, decode, models
from .train import RUNS_DIR


def load_cell(run_dir: str, domain: str, axis: str, tier: str):
    """Booster + decode state for one cell, or ``None`` when that cell was never fitted."""
    key = f"{domain}_{axis}_{tier}"
    mpath = os.path.join(run_dir, "models", f"{key}.json")
    dpath = os.path.join(run_dir, "models", f"{key}.decode.npz")
    if not (os.path.exists(mpath) and os.path.exists(dpath)):
        return None
    bst = xgb.Booster()
    bst.load_model(mpath)
    z = np.load(dpath, allow_pickle=False)
    return {
        "bst": bst, "trans": z["trans"], "prior": z["prior"], "cols": z["cols"],
        "stickiness": float(z["stickiness"]), "divide_prior": bool(z["divide_prior"]),
        "feat_names": [str(x) for x in z["feat_names"]], "key": key,
    }


def predict_table(cell, tabs: D.Tables, k: int, axis: str) -> np.ndarray:
    """Per-frame class indices for one cached table."""
    rows = np.flatnonzero(tabs.table_id == k)
    t = tabs.tables[k]
    proba = models.predict_proba(cell["bst"], tabs.X[np.ix_(rows, cell["cols"])],
                                 cell["feat_names"])
    post = decode.frame_posterior(t["frame_start"][tabs.win_idx[rows]],
                                  t["frame_end"][tabs.win_idx[rows]],
                                  proba, t["n_frames"], D.N_CLASSES[axis])
    if cell["stickiness"] <= 0.0:
        p = post
        if cell["divide_prior"]:
            p = p / np.clip(cell["prior"], decode.EPS, None)[None, :]
        return p.argmax(1)
    return decode.smooth(post, cell["trans"], cell["prior"],
                         divide_prior=cell["divide_prior"], stickiness=cell["stickiness"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tier", default="T1")
    ap.add_argument("--root", default=D.PINSORO_ROOT)
    ap.add_argument("--splits", nargs="*", default=["test"])
    ap.add_argument("--domains", nargs="*", default=["cc", "cr"])
    ap.add_argument("--axes", nargs="*", default=list(D.AXES))
    ap.add_argument("--out", default=None, help="default: runs/<run>/submission")
    ap.add_argument("--score", action="store_true",
                    help="also report per-frame kappa (annotated splits only)")
    args = ap.parse_args()

    run_dir = os.path.join(RUNS_DIR, args.run_name)
    if not os.path.isdir(run_dir):
        sys.exit(f"no such run: {run_dir}")
    out_root = args.out or os.path.join(run_dir, "submission")

    tabs = D.load(args.root, splits=tuple(args.splits), domains=tuple(args.domains))
    print(f"{len(tabs.tables)} tables over splits={args.splits}")

    written, scores = 0, {}
    for domain in args.domains:
        for axis in args.axes:
            cell = load_cell(run_dir, domain, axis, args.tier)
            if cell is None:
                print(f"  [skip] no fitted model for {domain}_{axis}_{args.tier}")
                continue
            true_all, pred_all = [], []
            for k, t in enumerate(tabs.tables):
                if t["domain"] != domain:
                    continue
                pred = predict_table(cell, tabs, k, axis)
                decode.write_prediction_csv(os.path.join(out_root, f"pinsoro-{domain}"),
                                            t["session"], t["role"], axis, pred,
                                            D.CLASS_NAMES[axis])
                written += 1
                if args.score:
                    lab = D.frame_labels(t, axis)
                    ok = lab >= 0
                    if ok.any():
                        true_all.append(lab[ok])
                        pred_all.append(pred[ok])
            if args.score and true_all:
                k_ = cohen_kappa(np.concatenate(true_all), np.concatenate(pred_all),
                                 num_classes=D.N_CLASSES[axis])
                scores[f"{domain}_{axis}"] = k_
                print(f"  {domain} {axis:7s} kappa {k_:.4f} "
                      f"({sum(len(x) for x in true_all)} frames)")

    print(f"wrote {written} prediction file(s) -> {out_root}")
    if scores:
        path = os.path.join(run_dir, f"scores_{'_'.join(args.splits)}_{args.tier}.json")
        with open(path, "w") as f:
            json.dump(scores, f, indent=1)
        print(f"scores -> {path}")


if __name__ == "__main__":
    main()
