#!/usr/bin/env python3
"""Grouped cross-validation and final fitting of the PInSoRo XGBoost baseline.

    python -m pinsoro_model.train --run-name simple01
    python -m pinsoro_model.train --tiers T0 --domains cr --folds 3 --no-wandb   # quick

One model per **cell** = (domain, axis, tier). cc and cr are fitted separately: their label
supports barely overlap (``cooperative`` never occurs in cr) and the partner is a child in one and
a robot in the other. Selection is by **per-frame** kappa after decoding, never window accuracy --
per-frame kappa is what the challenge scores.

Protocol, and why each piece is there:

* **GroupKFold by session.** Windows overlap 5x and the two children of a cc session share a room,
  cameras and a microphone, so any split finer than the session leaks.
* **Nested early stopping.** The stopping set is carved out of the *training* sessions of each
  fold; using the outer fold would make the reported kappa optimistic.
* **Two-stage search.** Tree parameters first at a fixed class-weight exponent, then the exponent
  at the best tree parameters. A full product would overfit the CV estimate itself -- there are
  only 27 cc and 17 cr sessions, and as few as 85 independent label runs (cr social).
* **Decode settings scored from cached posteriors.** Fitting a fold is expensive, scoring a decode
  setting on top of it is nearly free, so the whole decode grid is evaluated per configuration.
* **Out-of-fold pooling.** The headline kappa is computed once over all pooled out-of-fold frames,
  the way the challenge scores a submission; the per-fold spread is reported alongside because
  with this few sessions the mean alone is misleading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .kappa import cohen_kappa

from . import baselines, data as D, decode, models

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# `stickiness` tempers the HMM transition matrix, and it is load-bearing rather than cosmetic: the
# empirical 30 fps self-transition probability is ~0.9995, so an untempered chain charges ~11 nats
# to change class and collapses each session to a single label. The emissions are also far from
# independent -- consecutive frames are covered by the same overlapping 10 s windows -- so an
# untempered chain double-counts persistence the posterior already encodes. `stickiness = 0`
# disables smoothing entirely (plain argmax), which is the honest "no temporal model" reference.
STICKINESS_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
DIVIDE_PRIOR_GRID = (True, False)


# ── per-fold products ────────────────────────────────────────────────────────

def fold_posteriors(bst, tabs: D.Tables, rows: np.ndarray, cols: np.ndarray, axis: str,
                    feat_names, proba=None):
    """Per-frame posteriors for the held-out tables of one fold: ``[(posterior, labels), ...]``.

    Restricted to annotated frames, so unannotated stretches never enter a metric. ``proba`` lets a
    caller supply window probabilities it has already computed -- used by the bagged path, whose
    out-of-bag probabilities are averaged across a different subset of bags for every row.
    """
    if proba is None:
        proba = models.predict_proba(bst, tabs.X[np.ix_(rows, cols)], feat_names)
    n_classes = D.N_CLASSES[axis]
    tid, widx = tabs.table_id[rows], tabs.win_idx[rows]
    out = []
    for k in np.unique(tid):
        t = tabs.tables[k]
        lab = D.frame_labels(t, axis)
        ok = lab >= 0
        if not ok.any():
            continue
        sel = tid == k
        post = decode.frame_posterior(t["frame_start"][widx[sel]], t["frame_end"][widx[sel]],
                                      proba[sel], t["n_frames"], n_classes)
        out.append((post[ok].astype(np.float32), lab[ok]))
    return out


def fold_stats(tabs: D.Tables, rows: np.ndarray, axis: str, n_classes: int):
    """Transition matrix and class prior estimated from the given rows' tables (training only)."""
    seqs = [D.frame_labels(tabs.tables[k], axis) for k in np.unique(tabs.table_id[rows])]
    return decode.transition_matrix(seqs, n_classes), decode.class_prior(seqs, n_classes)


def score_decode(folds, n_classes: int, stickiness: float, divide_prior: bool) -> dict:
    """Pooled and per-fold kappa for one decode setting, over cached fold posteriors."""
    pooled_t, pooled_p, per_fold = [], [], []
    for f in folds:
        ft, fp = [], []
        for post, lab in f["posts"]:
            p = post.astype(np.float64)
            if stickiness <= 0.0:
                if divide_prior:
                    p = p / np.clip(f["prior"], decode.EPS, None)[None, :]
                pred = p.argmax(1)
            else:
                pred = decode.smooth(p, f["trans"], f["prior"],
                                     divide_prior=divide_prior, stickiness=stickiness)
            ft.append(lab)
            fp.append(pred)
        if not ft:
            continue
        ft, fp = np.concatenate(ft), np.concatenate(fp)
        per_fold.append(cohen_kappa(ft, fp, num_classes=n_classes))
        pooled_t.append(ft)
        pooled_p.append(fp)
    if not pooled_t:
        return {"kappa": float("nan"), "fold_mean": float("nan"),
                "fold_std": float("nan"), "folds": []}
    a = np.asarray(per_fold, float)
    return {"kappa": cohen_kappa(np.concatenate(pooled_t), np.concatenate(pooled_p),
                                 num_classes=n_classes),
            "fold_mean": float(np.nanmean(a)), "fold_std": float(np.nanstd(a)),
            "folds": [float(x) for x in a]}


# ── one cross-validated configuration ────────────────────────────────────────

def run_cv(tabs: D.Tables, cell_rows: np.ndarray, cols: np.ndarray, axis: str,
           params: dict, alpha: float, n_folds: int, seed: int, feat_names) -> dict:
    """GroupKFold CV of one configuration, scored over the whole decode grid."""
    n_classes = D.N_CLASSES[axis]
    keep = cell_rows[tabs.y[axis][cell_rows] >= 0]
    groups = tabs.group[keep]
    n_splits = min(n_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    folds, rounds = [], []
    for tr_i, va_i in gkf.split(keep, groups=groups):
        tr, va = keep[tr_i], keep[va_i]
        tr_groups = tabs.group[tr]
        if len(np.unique(tr_groups)) >= 4:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            fit_i, es_i = next(gss.split(tr, groups=tr_groups))
            fit_rows, es_rows = tr[fit_i], tr[es_i]
        else:
            fit_rows, es_rows = tr, tr

        yf = tabs.y[axis][fit_rows]
        wf = D.sample_weights(yf, tabs.purity[axis][fit_rows], n_classes, alpha)
        ye = tabs.y[axis][es_rows]
        we = D.sample_weights(ye, tabs.purity[axis][es_rows], n_classes, alpha)
        bst = models.fit(tabs.X[np.ix_(fit_rows, cols)], yf, wf, n_classes, params,
                         feature_names=feat_names,
                         eval_set=(tabs.X[np.ix_(es_rows, cols)], ye, we), seed=seed)
        rounds.append(int(getattr(bst, "best_iteration", bst.num_boosted_rounds() - 1)) + 1)

        trans, prior = fold_stats(tabs, tr, axis, n_classes)
        folds.append({"posts": fold_posteriors(bst, tabs, va, cols, axis, feat_names),
                      "trans": trans, "prior": prior})

    raw = score_decode(folds, n_classes, 0.0, False)
    best, best_key, grid = raw, (0.0, False), {}
    for st in STICKINESS_GRID:
        for dp in DIVIDE_PRIOR_GRID:
            r = raw if (st == 0.0 and dp is False) else score_decode(folds, n_classes, st, dp)
            grid[f"st{st}_dp{int(dp)}"] = r["kappa"]
            if np.isfinite(r["kappa"]) and r["kappa"] > best["kappa"]:
                best, best_key = r, (st, dp)

    return {
        "n_folds": n_splits, "best_rounds": rounds,
        "median_rounds": int(np.median(rounds)) if rounds else models.MAX_ROUNDS,
        "kappa_raw": raw["kappa"], "kappa_decoded": best["kappa"],
        "fold_mean": best["fold_mean"], "fold_std": best["fold_std"], "folds": best["folds"],
        "stickiness": float(best_key[0]), "divide_prior": bool(best_key[1]),
        "decode_grid": grid,
    }


# ── the sweep ────────────────────────────────────────────────────────────────

def search_cell(tabs, cell_rows, cols, axis, n_folds, seed, feat_names, log, n_configs, alphas):
    """Two-stage search: tree parameters at a fixed alpha, then alpha at the best parameters."""
    trials, best = [], None
    grid = models.PARAM_GRID[:n_configs]
    for i, over in enumerate(grid):
        t0 = time.time()
        r = run_cv(tabs, cell_rows, cols, axis, over, 0.5, n_folds, seed, feat_names)
        r.update({"stage": "params", "params": over, "alpha": 0.5, "secs": time.time() - t0})
        trials.append(r)
        log("    params {}/{} {} -> kappa {:.4f} (raw {:.4f}) [{:.0f}s]".format(
            i + 1, len(grid), over, r["kappa_decoded"], r["kappa_raw"], r["secs"]))
        if best is None or r["kappa_decoded"] > best["kappa_decoded"]:
            best = r
    for a in alphas:
        if a == 0.5:
            continue
        t0 = time.time()
        r = run_cv(tabs, cell_rows, cols, axis, best["params"], a, n_folds, seed, feat_names)
        r.update({"stage": "alpha", "params": best["params"], "alpha": a,
                  "secs": time.time() - t0})
        trials.append(r)
        log("    alpha {} -> kappa {:.4f} (raw {:.4f}) [{:.0f}s]".format(
            a, r["kappa_decoded"], r["kappa_raw"], r["secs"]))
        if r["kappa_decoded"] > best["kappa_decoded"]:
            best = r
    return best, trials


def fit_final(tabs, cell_rows, cols, axis, best, seed, feat_names):
    """Refit on every session of the cell at the selected configuration."""
    n_classes = D.N_CLASSES[axis]
    keep = cell_rows[tabs.y[axis][cell_rows] >= 0]
    w = D.sample_weights(tabs.y[axis][keep], tabs.purity[axis][keep], n_classes, best["alpha"])
    bst = models.fit(tabs.X[np.ix_(keep, cols)], tabs.y[axis][keep], w, n_classes,
                     best["params"], feature_names=feat_names,
                     num_boost_round=best["median_rounds"], seed=seed)
    trans, prior = fold_stats(tabs, keep, axis, n_classes)
    return bst, trans, prior


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=D.PINSORO_ROOT)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--domains", nargs="*", default=["cc", "cr"])
    ap.add_argument("--axes", nargs="*", default=list(D.AXES))
    ap.add_argument("--tiers", nargs="*", default=["T0", "T1", "T2"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param-configs", type=int, default=len(models.PARAM_GRID),
                    help="how many entries of models.PARAM_GRID to try")
    ap.add_argument("--alphas", nargs="*", type=float, default=models.ALPHA_GRID)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="multimediate26-comp_model")
    ap.add_argument("--drop-features", default=None,
                    help="regex; matching features are removed from every tier. Use it to ablate "
                         "the session-coded metadata (e.g. '^meta\\.(age|gender)$') -- those are "
                         "constant within a session, so they carry recording identity. GroupKFold "
                         "already blocks memorising a held-out session, but the ablation shows how "
                         "much the model leans on them.")
    ap.add_argument("--top-features", type=int, default=30)
    args = ap.parse_args()

    run_name = args.run_name or f"simple-xgb-{date.today().isoformat()}"
    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    log_fh = open(os.path.join(out_dir, "train.log"), "a")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    tabs = D.load(args.root, splits=tuple(args.splits), domains=tuple(args.domains))
    schema = D.load_schema()
    log(f"run {run_name}: {len(tabs)} windows, {len(tabs.tables)} tables, "
        f"{len(tabs.feat_names)} features (schema {schema['schema_hash'][:12]})")

    base = {f"{d}_{a}": baselines.evaluate(tabs, d, a) for d in args.domains for a in args.axes}
    for k, v in base.items():
        if v:
            log(f"baseline {k:12s} majority {v['majority']:+.4f}  "
                f"session-mode oracle {v['session_mode_oracle']:.4f}")

    wb = _init_wandb(args, run_name, out_dir, tabs, schema, base, log) if args.wandb else None

    results = {}
    for domain in args.domains:
        for axis in args.axes:
            cell_rows = np.flatnonzero(tabs.mask(domain=domain))
            if cell_rows.size == 0:
                continue
            for tier in args.tiers:
                cols = D.tier_columns(tabs.feat_names, tier, schema)
                if args.drop_features:
                    pat = re.compile(args.drop_features)
                    cols = np.array([i for i in cols if not pat.search(tabs.feat_names[i])],
                                    dtype=np.int64)
                    if cols.size == 0:
                        log(f"   [skip] --drop-features removed every {tier} column")
                        continue
                names = [tabs.feat_names[i] for i in cols]
                key = f"{domain}_{axis}_{tier}"
                log(f"\n== {key}: {cell_rows.size} windows, "
                    f"{len(np.unique(tabs.group[cell_rows]))} sessions, {len(cols)} features")
                t0 = time.time()
                best, trials = search_cell(tabs, cell_rows, cols, axis, args.folds, args.seed,
                                           names, log, args.param_configs, args.alphas)
                bst, trans, prior = fit_final(tabs, cell_rows, cols, axis, best, args.seed, names)
                imp = models.gain_importance(bst, names)[:args.top_features]

                mpath = os.path.join(out_dir, "models", f"{key}.json")
                bst.save_model(mpath)
                np.savez(os.path.join(out_dir, "models", f"{key}.decode.npz"),
                         trans=trans, prior=prior, cols=cols,
                         stickiness=np.float64(best["stickiness"]),
                         divide_prior=bool(best["divide_prior"]),
                         feat_names=np.array(names, dtype=np.str_))
                results[key] = {
                    "domain": domain, "axis": axis, "tier": tier,
                    "n_features": len(cols), "n_windows": int(cell_rows.size),
                    "n_sessions": int(len(np.unique(tabs.group[cell_rows]))),
                    "best": best, "trials": trials, "top_features": imp,
                    "secs": time.time() - t0, "model": mpath,
                }
                log("   BEST {}: kappa {:.4f} (raw {:.4f})  fold {:.4f}+-{:.4f}  "
                    "alpha {}  stickiness {} divide_prior {}  {}  [{:.0f}s]".format(
                        key, best["kappa_decoded"], best["kappa_raw"], best["fold_mean"],
                        best["fold_std"], best["alpha"], best["stickiness"],
                        best["divide_prior"], best["params"], time.time() - t0))
                log("   top features: " + ", ".join(n for n, _ in imp[:8]))
                if wb:
                    _log_cell(wb, key, results[key], base.get(f"{domain}_{axis}", {}))

    with open(os.path.join(out_dir, "cv_results.json"), "w") as f:
        json.dump({"args": vars(args), "baselines": base, "results": results}, f, indent=1,
                  default=float)
    _write_summary(out_dir, results, base, log)
    if wb:
        _log_summary(wb, results, base)
        wb.finish()
    log(f"\nartifacts -> {out_dir}")
    log_fh.close()


# ── reporting ────────────────────────────────────────────────────────────────

def _write_summary(out_dir, results, base, log):
    rows = ["domain,axis,tier,n_features,kappa_raw,kappa_decoded,fold_mean,fold_std,"
            "alpha,stickiness,divide_prior,majority,session_mode_oracle"]
    for r in results.values():
        b = base.get(f"{r['domain']}_{r['axis']}", {})
        bt = r["best"]
        rows.append("{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{:.4f},{:.4f}".format(
            r["domain"], r["axis"], r["tier"], r["n_features"], bt["kappa_raw"],
            bt["kappa_decoded"], bt["fold_mean"], bt["fold_std"], bt["alpha"],
            bt["stickiness"], int(bt["divide_prior"]),
            b.get("majority", float("nan")), b.get("session_mode_oracle", float("nan"))))
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    log("\n" + "\n".join(rows))


def _has_wandb_credentials() -> bool:
    """W&B credentials present: env key or a netrc login entry."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        with open(os.path.expanduser("~/.netrc")) as f:
            return "api.wandb.ai" in f.read()
    except OSError:
        return False


def _init_wandb(args, run_name, out_dir, tabs, schema, base, log):
    try:
        import wandb
    except ImportError:
        log("[wandb] not installed -- continuing without it")
        return None
    if not _has_wandb_credentials():
        log("[wandb] no credentials (set WANDB_API_KEY or run `wandb login`) -- skipping")
        return None
    wdir = os.path.join(out_dir, "wandb")
    os.makedirs(wdir, exist_ok=True)
    cfg = dict(vars(args))
    cfg.update({
        "schema_hash": schema["schema_hash"], "n_windows": len(tabs),
        "n_tables": len(tabs.tables), "tier_counts": schema["counts"],
        "window_sec": schema["window_sec"], "hop_sec": schema["hop_sec"],
        "model": "xgboost", "xgboost_version": models.xgb.__version__,
        "stickiness_grid": list(STICKINESS_GRID),
    })
    run = wandb.init(project=args.wandb_project, name=run_name, dir=wdir,
                     group="pinsoro-simple", job_type="cv", config=cfg,
                     tags=["pinsoro", "simple", "xgboost"])
    for k, v in base.items():
        if v:
            run.summary[f"baseline/{k}/majority"] = v["majority"]
            run.summary[f"baseline/{k}/session_mode_oracle"] = v["session_mode_oracle"]
    log(f"[wandb] {run.url}")
    return run


def _log_cell(wb, key, res, base):
    bt = res["best"]
    metrics = {
        f"cv/{key}/kappa_decoded": bt["kappa_decoded"],
        f"cv/{key}/kappa_raw": bt["kappa_raw"],
        f"cv/{key}/fold_mean": bt["fold_mean"],
        f"cv/{key}/fold_std": bt["fold_std"],
        f"cv/{key}/alpha": bt["alpha"],
        f"cv/{key}/stickiness": bt["stickiness"],
        f"cv/{key}/n_features": res["n_features"],
        f"cv/{key}/secs": res["secs"],
    }
    wb.log(metrics)
    for k, v in metrics.items():
        wb.summary[k] = v
    if base:
        wb.summary[f"cv/{key}/vs_session_mode_oracle"] = (
            bt["kappa_decoded"] - base["session_mode_oracle"])


def _log_summary(wb, results, base):
    import wandb
    cols = ["domain", "axis", "tier", "n_features", "kappa_raw", "kappa_decoded",
            "fold_mean", "fold_std", "alpha", "stickiness", "session_mode_oracle"]
    rows = []
    for r in results.values():
        bt, b = r["best"], base.get(f"{r['domain']}_{r['axis']}", {})
        rows.append([r["domain"], r["axis"], r["tier"], r["n_features"], bt["kappa_raw"],
                     bt["kappa_decoded"], bt["fold_mean"], bt["fold_std"], bt["alpha"],
                     bt["stickiness"], b.get("session_mode_oracle", float("nan"))])
    wb.log({"summary": wandb.Table(columns=cols, data=rows)})
    for r in results.values():
        wb.log({f"importance/{r['domain']}_{r['axis']}_{r['tier']}":
                wandb.Table(columns=["feature", "total_gain"],
                            data=[[n, g] for n, g in r["top_features"]])})


if __name__ == "__main__":
    main()
