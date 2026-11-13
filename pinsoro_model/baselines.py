"""Reference points that make a kappa number interpretable.

Neither is a fitted model -- they exist so the XGBoost numbers can be read against something:

* **majority** -- always predict the most frequent class. Unweighted kappa is 0 by construction,
  so this only confirms the scorer is wired correctly.
* **session-mode oracle** -- predict each session's *own* modal class for all of its frames. It
  peeks at the held-out labels, so it is not a usable model; it is the ceiling for "do nothing but
  recognise which recording this is". If a trained model lands near it, that model is mostly doing
  session identification rather than reading behaviour, which is the central question this whole
  baseline exists to answer.
"""

from __future__ import annotations

import numpy as np

from .kappa import cohen_kappa

from . import data as D


def evaluate(tabs: D.Tables, domain: str, axis: str) -> dict:
    """Per-frame kappa of both reference predictors over every table of one (domain, axis)."""
    n_classes = D.N_CLASSES[axis]
    true_all, maj_all, mode_all = [], [], []

    counts = np.zeros(n_classes, np.int64)
    per_table = []
    for k, t in enumerate(tabs.tables):
        if t["domain"] != domain:
            continue
        lab = D.frame_labels(t, axis)
        v = lab[lab >= 0]
        if v.size == 0:
            continue
        counts += np.bincount(v, minlength=n_classes)
        per_table.append((lab, int(np.bincount(v, minlength=n_classes).argmax())))

    if not per_table:
        return {}
    majority = int(counts.argmax())
    for lab, sess_mode in per_table:
        ok = lab >= 0
        true_all.append(lab[ok])
        maj_all.append(np.full(ok.sum(), majority))
        mode_all.append(np.full(ok.sum(), sess_mode))

    t = np.concatenate(true_all)
    return {
        "majority": cohen_kappa(t, np.concatenate(maj_all), num_classes=n_classes),
        "session_mode_oracle": cohen_kappa(t, np.concatenate(mode_all), num_classes=n_classes),
        "n_frames": int(t.size),
        "n_tables": len(per_table),
    }


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=D.PINSORO_ROOT)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--out", default=None, help="optional JSON path")
    args = ap.parse_args()

    tabs = D.load(args.root, splits=tuple(args.splits))
    out = {}
    for domain in ("cc", "cr"):
        for axis in D.AXES:
            r = evaluate(tabs, domain, axis)
            if not r:
                continue
            out[f"{domain}_{axis}"] = r
            print(f"{domain} {axis:7s}  majority {r['majority']:+.4f}   "
                  f"session-mode oracle {r['session_mode_oracle']:.4f}   "
                  f"({r['n_tables']} tables, {r['n_frames']} frames)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
