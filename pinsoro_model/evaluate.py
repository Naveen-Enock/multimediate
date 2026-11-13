#!/usr/bin/env python3
"""Train on PInSoRo's own train split, score its own val split. No cross-validation.

    python -m pinsoro_model.evaluate --feature-set V2 --run-name v2-01
    python -m pinsoro_model.evaluate --feature-set T2 --run-name val01 --no-wandb

Reports unweighted per-frame Cohen's kappa as --
``kappa_cc_social``, ``kappa_cc_task``, ``kappa_cr_social``, ``kappa_cr_task``, ``kappa_mean`` -- one
set per feature set. Also reports per-class recall and the confusion matrix, since kappa alone hides
what happens to rare classes.

Feature sets: ``T0``/``T1``/``T2`` are tiers of the original 442-feature tables
(``simple_feats.npz``); ``V2`` is the curated set in ``feature_sets.py`` (``v2_feats.npz``), 49
features for cc and 39 for cr.

**Nothing is selected on val.** The boosting round count and the decode setting are both chosen on a
grouped slice held out of *train*. The model is then refitted on all of train and val is scored once.
Splits are PInSoRo's on-disk ``{train,val}-{cc,cr}`` directories as shipped. ``test-*`` has no
annotations and is never scored here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .kappa import cohen_kappa

from . import bagging, data as D, decode, linear_models, models, select, stacking
from .train import RUNS_DIR, STICKINESS_GRID, fold_posteriors, fold_stats, _has_wandb_credentials
from .windows import run_lengths

CELLS = [("cc", "social"), ("cc", "task"), ("cr", "social"), ("cr", "task")]
FEATURE_SETS = ("T0", "T1", "T2", "V2")


def _decode(posts, trans, prior, stickiness):
    """Concatenated ``(y_true, y_pred)`` over a list of per-table ``(posterior, labels)``."""
    t, p = [], []
    for post, lab in posts:
        q = post.astype(np.float64)
        pred = (q.argmax(1) if stickiness <= 0.0
                else decode.smooth(q, trans, prior, divide_prior=False, stickiness=stickiness))
        t.append(lab)
        p.append(pred)
    if not t:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(t).astype(np.int64), np.concatenate(p).astype(np.int64)


def _kappa(posts, trans, prior, stickiness, n_classes):
    t, p = _decode(posts, trans, prior, stickiness)
    return cohen_kappa(t, p, num_classes=n_classes) if t.size else float("nan")


def per_class(y_true, y_pred, n_classes):
    """Confusion matrix (rows = truth), per-class support and recall."""
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    support = cm.sum(axis=1)
    recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return cm, support, recall


def fit_cell(train_tabs, domain, axis, feature_set, schema, alpha, seed, params, drop=None,
             model: str = "xgb", rows=None):
    """Fit one (domain, axis) model on train; everything needed to predict, or None if empty.

    The boosting round count and the decode setting are chosen on a grouped slice held out of
    train, then the model is refitted on all of train.

    ``rows`` restricts training to a subset of the cell's rows, which is how a cross-validation
    fold trains on its own sessions only. ``None`` uses every labelled row of the cell.
    """
    n_classes = D.N_CLASSES[axis]
    cols = D.columns_for(train_tabs.feat_names, feature_set, domain, schema)
    if drop:
        pat = re.compile(drop)
        cols = np.array([i for i in cols if not pat.search(train_tabs.feat_names[i])], np.int64)
    names = [train_tabs.feat_names[i] for i in cols]
    tr_all = np.flatnonzero(train_tabs.mask(domain=domain) & (train_tabs.y[axis] >= 0))
    if rows is not None:
        tr_all = np.intersect1d(tr_all, np.asarray(rows, np.int64))
    if tr_all.size == 0:
        return None
    if model in linear_models.SUBSAMPLE_CC and domain == "cc":
        # Keep every 5th window: exactly non-overlapping at a 10 s window / 2 s hop. See
        # linear_models.SUBSAMPLE_CC for why only the kernel models need it, and only on cc.
        tr_all = tr_all[train_tabs.win_idx[tr_all] % 5 == 0]

    # ── selection stage: a grouped slice of train, so val stays untouched ────
    groups = train_tabs.group[tr_all]
    if len(np.unique(groups)) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        a, b = next(gss.split(tr_all, groups=groups))
        fit_rows, dev_rows = tr_all[a], tr_all[b]
    else:
        fit_rows = dev_rows = tr_all

    w_fit = D.sample_weights(train_tabs.y[axis][fit_rows], train_tabs.purity[axis][fit_rows],
                             n_classes, alpha)
    w_dev = D.sample_weights(train_tabs.y[axis][dev_rows], train_tabs.purity[axis][dev_rows],
                             n_classes, alpha)
    tr_trans, tr_prior = fold_stats(train_tabs, fit_rows, axis, n_classes)
    X_fit, y_fit = train_tabs.X[np.ix_(fit_rows, cols)], train_tabs.y[axis][fit_rows]
    hp, oof, full_nb, aug_names = None, None, None, None
    if model == "xgb_nb":
        # NB's smoothing is picked on the dev slice, then held fixed inside every out-of-fold fit.
        best_nb = None
        for cand in linear_models.GRID["gaussian_nb"]:
            nb = linear_models.fit(X_fit, y_fit, w_fit, n_classes, "gaussian_nb", cand, seed)
            k = _kappa(fold_posteriors(nb, train_tabs, dev_rows, cols, axis, names),
                       tr_trans, tr_prior, 0.0, n_classes)
            if best_nb is None or (np.isfinite(k) and k > best_nb[0]):
                best_nb = (k if np.isfinite(k) else -9.0, cand)
        nb_params = best_nb[1]
        oof, full_nb = stacking.nb_oof_features(train_tabs, tr_all, cols, axis, alpha,
                                                nb_params, seed)
        aug_names = names + stacking.nb_feature_names(axis)
        pos = {int(r): i for i, r in enumerate(tr_all)}
        i_fit = np.array([pos[int(r)] for r in fit_rows], np.int64)
        i_dev = np.array([pos[int(r)] for r in dev_rows], np.int64)
        Xf = np.hstack([X_fit, oof[i_fit]])
        Xd = np.hstack([train_tabs.X[np.ix_(dev_rows, cols)], oof[i_dev]])
        probe_bst = models.fit(Xf, y_fit, w_fit, n_classes, params, feature_names=aug_names,
                               eval_set=(Xd, train_tabs.y[axis][dev_rows], w_dev), seed=seed)
        rounds = int(getattr(probe_bst, "best_iteration",
                             probe_bst.num_boosted_rounds() - 1)) + 1
        # Score the dev slice from its *out-of-fold* NB columns. Routing it through the wrapper
        # would use the full-train NB, which has seen these very sessions.
        dev_posts = fold_posteriors(None, train_tabs, dev_rows, cols, axis, names,
                                    proba=models.predict_proba(probe_bst, Xd, aug_names))
        hp = dict(nb_params, n_features=len(aug_names))
    elif model == "xgb":
        probe = models.fit(X_fit, y_fit, w_fit, n_classes, params, feature_names=names,
                           eval_set=(train_tabs.X[np.ix_(dev_rows, cols)],
                                     train_tabs.y[axis][dev_rows], w_dev), seed=seed)
        rounds = int(getattr(probe, "best_iteration", probe.num_boosted_rounds() - 1)) + 1
        dev_posts = fold_posteriors(probe, train_tabs, dev_rows, cols, axis, names)
    else:
        # The linear models have no early stopping, so the same grouped dev slice selects their one
        # hyperparameter instead, scored by decoded kappa. Still nothing from val.
        rounds, best = 0, None
        for cand in linear_models.GRID[model]:
            mdl = linear_models.fit(X_fit, y_fit, w_fit, n_classes, model, cand, seed)
            posts = fold_posteriors(mdl, train_tabs, dev_rows, cols, axis, names)
            k = _kappa(posts, tr_trans, tr_prior, 0.0, n_classes)
            if best is None or (np.isfinite(k) and k > best[0]):
                best = (k if np.isfinite(k) else -9.0, cand, posts)
        _, hp, dev_posts = best
    dev_grid = {float(s): _kappa(dev_posts, tr_trans, tr_prior, s, n_classes)
                for s in STICKINESS_GRID}
    stickiness = max(dev_grid, key=lambda s: dev_grid[s] if np.isfinite(dev_grid[s]) else -9)

    # ── final fit on all of train ────────────────────────────────────────────
    w_all = D.sample_weights(train_tabs.y[axis][tr_all], train_tabs.purity[axis][tr_all],
                             n_classes, alpha)
    X_all, y_all = train_tabs.X[np.ix_(tr_all, cols)], train_tabs.y[axis][tr_all]
    if model == "xgb":
        bst = models.fit(X_all, y_all, w_all, n_classes, params, feature_names=names,
                         num_boost_round=rounds, seed=seed)
    elif model == "xgb_nb":
        bst = stacking.NBAugmentedBooster(
            models.fit(np.hstack([X_all, oof]), y_all, w_all, n_classes, params,
                       feature_names=aug_names, num_boost_round=rounds, seed=seed),
            full_nb, n_classes, aug_names)
    else:
        bst = linear_models.fit(X_all, y_all, w_all, n_classes, model, hp, seed)
    trans, prior = fold_stats(train_tabs, tr_all, axis, n_classes)
    return {"bst": bst, "cols": cols, "names": names, "trans": trans, "prior": prior,
            "stickiness": float(stickiness), "rounds": rounds, "dev_grid": dev_grid,
            "model": model, "hp": hp,
            "n_train_windows": int(tr_all.size),
            "n_train_sessions": int(len(np.unique(train_tabs.group[tr_all])))}


def predict_table_frames(m, tabs, k, axis):
    """Decoded prediction for every frame of table ``k``, annotated or not.

    For splits without annotations (test). Scoring against labels goes through
    ``fold_posteriors`` + ``_decode`` instead, which smooths over annotated frames only -- that is the
    path every reported kappa uses.
    """
    rows = np.flatnonzero(tabs.table_id == k)
    t = tabs.tables[k]
    proba = models.predict_proba(m["bst"], tabs.X[np.ix_(rows, m["cols"])], m["names"])
    post = decode.frame_posterior(t["frame_start"][tabs.win_idx[rows]],
                                  t["frame_end"][tabs.win_idx[rows]],
                                  proba, t["n_frames"], D.N_CLASSES[axis])
    if m["stickiness"] <= 0.0:
        return post.argmax(1)
    return decode.smooth(post, m["trans"], m["prior"], divide_prior=False,
                         stickiness=m["stickiness"])


def _kappa_from_cm(cm):
    """Unweighted Cohen's kappa straight from a confusion matrix (same formula as the metric)."""
    n = cm.sum()
    if n <= 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = float(((cm.sum(1) / n) * (cm.sum(0) / n)).sum())
    return 1.0 if abs(1.0 - pe) < 1e-12 else float((po - pe) / (1.0 - pe))


def bootstrap_ci(posts, trans, prior, stickiness, n_classes, n_boot=2000, seed=0):
    """Percentile CI for the val kappa, resampling the val **children** with replacement.

    The children are the independent units, so this is what the reported number's uncertainty
    actually rests on -- cr social has 4 annotated children. Each child's confusion matrix is
    computed once and the bootstrap just sums them, so 2000 resamples are cheap.
    """
    cms = []
    for post, lab in posts:
        t, p = _decode([(post, lab)], trans, prior, stickiness)
        cm = np.zeros((n_classes, n_classes), np.int64)
        np.add.at(cm, (t, p), 1)
        cms.append(cm)
    if len(cms) < 2:
        return None
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cms), len(cms))
        k = _kappa_from_cm(sum(cms[i] for i in idx))
        if np.isfinite(k):
            ks.append(k)
    return [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))] if ks else None


def eval_cell(train_tabs, val_tabs, domain, axis, feature_set, schema, alpha, seed, params,
              drop=None, bags: int = 0, model: str = "xgb"):
    """Fit on train, score val once. Round count and decode setting come from a train-only slice."""
    n_classes = D.N_CLASSES[axis]
    va = np.flatnonzero(val_tabs.mask(domain=domain) & (val_tabs.y[axis] >= 0))
    if va.size == 0:
        return None
    if bags > 0 and model != "xgb":
        raise ValueError("--bags is implemented for the xgb path only")
    m = (bagging.fit_bagged_cell(train_tabs, domain, axis, feature_set, schema, alpha, seed,
                                 params, bags, drop)
         if bags > 0 else
         fit_cell(train_tabs, domain, axis, feature_set, schema, alpha, seed, params, drop,
                  model=model))
    if m is None:
        return None
    val_posts = fold_posteriors(m["bst"], val_tabs, va, m["cols"], axis, m["names"])
    if not val_posts:
        return None

    y_true, y_pred = _decode(val_posts, m["trans"], m["prior"], m["stickiness"])
    kappa = cohen_kappa(y_true, y_pred, num_classes=n_classes)
    raw = _kappa(val_posts, m["trans"], m["prior"], 0.0, n_classes)
    cm, support, recall = per_class(y_true, y_pred, n_classes)
    ci = bootstrap_ci(val_posts, m["trans"], m["prior"], m["stickiness"], n_classes, seed=seed)
    curve = bagging.bag_curve(m, val_tabs, va, axis, n_classes) if bags > 0 else {}

    # Effective sample size on val: independent label runs, not frames.
    n_runs = 0
    for k in np.unique(val_tabs.table_id[va]):
        n_runs += len(run_lengths(D.frame_labels(val_tabs.tables[k], axis))[1])

    return {
        "kappa": float(kappa), "kappa_raw": float(raw), "stickiness": m["stickiness"],
        "rounds": m["rounds"], "alpha": alpha, "n_features": int(len(m["cols"])),
        "n_train_windows": m["n_train_windows"], "n_val_windows": int(va.size),
        "n_train_sessions": m["n_train_sessions"],
        "n_val_sessions": int(len(np.unique(val_tabs.group[va]))),
        "val_frames": int(y_true.size), "val_label_runs": n_runs,
        "classes": D.CLASS_NAMES[axis],
        "confusion": cm.tolist(), "support": support.tolist(),
        "recall": [None if not np.isfinite(r) else float(r) for r in recall],
        "true_share": (support / max(support.sum(), 1)).tolist(),
        "pred_share": (np.bincount(y_pred, minlength=n_classes) / max(y_pred.size, 1)).tolist(),
        "dev_stickiness_grid": {str(k): float(v) for k, v in m["dev_grid"].items()},
        "kappa_ci95": ci, "bags": int(m.get("bags", 0)), "oob_kappa": m.get("oob_kappa"),
        "model": m.get("model", "xgb"), "hp": m.get("hp"),
        "bag_curve": curve, "mean_oob_sessions": m.get("mean_oob_sessions"),
        "params": params,
        "top_features": models.gain_importance(m["bst"], m["names"])[:20],
        "_y_true": y_true, "_y_pred": y_pred,
    }


def _init_wandb(args, out_dir, counts):
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed -- skipping")
        return None
    if not _has_wandb_credentials():
        print("[wandb] no credentials -- skipping")
        return None
    run = wandb.init(project=args.wandb_project, name=args.run_name, dir=out_dir,
                     group="pinsoro-simple", job_type="train-val",
                     config=dict(vars(args), feature_counts=counts,
                                 xgboost_version=models.xgb.__version__),
                     tags=["pinsoro", "simple", "train-val"] + list(args.model)
                          + list(args.feature_set))
    print(f"[wandb] {run.url}")
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=D.PINSORO_ROOT)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--feature-set", nargs="*", default=["V2"], choices=FEATURE_SETS)
    ap.add_argument("--model", nargs="*", default=["xgb"],
                    choices=("xgb", "xgb_nb") + linear_models.KINDS,
                    help="xgb = gradient-boosted trees; linreg = ridge on the one-hot indicator "
                         "matrix (literal linear regression); logreg = multinomial logistic "
                         "regression")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-features", default=None)
    ap.add_argument("--composite", action="store_true",
                    help="choose one model family per cell by grouped CV inside train (see "
                         "select.py), then score val once with the chosen models")
    ap.add_argument("--candidates", nargs="*", default=["xgb", "gaussian_nb"],
                    help="families the composite may choose between")
    ap.add_argument("--bags", type=int, default=0,
                    help="classic session-level bootstrap: number of bagged members (0 = off)")
    ap.add_argument("--run-name", default="v2-01")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="multimediate26-comp_model")
    args = ap.parse_args()

    out_dir = os.path.join(RUNS_DIR, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    schema = D.load_schema()

    tabs = {}
    for fs in args.feature_set:
        sfx = D.FEATURE_SET_SUFFIX[fs]
        if sfx not in tabs:
            tabs[sfx] = (D.load(args.root, splits=(args.train_split,), suffix=sfx),
                         D.load(args.root, splits=(args.val_split,), suffix=sfx))
            tr, va = tabs[sfx]
            print(f"{sfx}: train {len(tr)} windows / {len(tr.tables)} tables   "
                  f"val {len(va)} windows / {len(va.tables)} tables")
    counts = {fs: {d: int(len(D.columns_for(tabs[D.FEATURE_SET_SUFFIX[fs]][0].feat_names,
                                           fs, d, schema))) for d in ("cc", "cr")}
              for fs in args.feature_set}
    print("feature counts:", counts)

    wb = _init_wandb(args, out_dir, counts) if args.wandb else None

    results, vectors, labels, label_fs = {}, {}, [], {}
    for mdl in args.model:
        for fs in args.feature_set:
            label = ("composite" if args.composite else
                     (mdl if len(args.feature_set) == 1 else f"{mdl}|{fs}"))
            labels.append(label)
            label_fs[label] = fs
            train_tabs, val_tabs = tabs[D.FEATURE_SET_SUFFIX[fs]]
            for domain, axis in CELLS:
                use, sel = mdl, None
                if args.composite:
                    print(f"  selecting for {domain}_{axis} by grouped CV inside train:")
                    sel = select.choose(train_tabs, domain, axis, fs, schema, args.alpha,
                                        args.seed, models.PARAM_GRID[3],
                                        candidates=tuple(args.candidates))
                    use = sel["model"]
                r = eval_cell(train_tabs, val_tabs, domain, axis, fs, schema, args.alpha,
                              args.seed, models.PARAM_GRID[3], args.drop_features,
                              bags=args.bags, model=use)
                if r is None:
                    continue
                if sel is not None:
                    r["selected_model"] = use
                    r["selection"] = sel
                key = f"{label}/kappa_{domain}_{axis}"
                vectors[key] = (r.pop("_y_true"), r.pop("_y_pred"))
                results[key] = r
                ci = r.get("kappa_ci95")
                extra = (f", {r['bags']} bags, OOB kappa {r['oob_kappa']:+.4f}"
                         if r.get("bags") else "")
                fitdesc = ((f"{r['rounds']} rounds" if r.get("rounds") else "")
                           + (f" {r['hp']}" if r.get("hp") else "")).strip()
                tag = f"{label}[{r['selected_model']}]" if r.get("selected_model") else label
                print(f"  {tag} kappa_{domain}_{axis:<7} = {r['kappa']:+.4f}  "
                      f"(raw {r['kappa_raw']:+.4f}"
                      + (f", 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                      + f", {r['n_features']} feats, {fitdesc}, "
                      f"stickiness {r['stickiness']}, {r['val_label_runs']} val label runs{extra})")
                if r.get("bag_curve"):
                    print("      kappa vs bags: "
                          + "  ".join(f"B={b} {v:+.4f}" for b, v in r["bag_curve"].items()))
                print("      recall  " + "  ".join(
                    f"{c[:11]} {('%.2f' % rc) if rc is not None else ' n/a'} (n={s_})"
                    for c, rc, s_ in zip(r["classes"], r["recall"], r["support"])))

    print(f"\n{'model':<8} {'kappa_cc_social':>16} {'kappa_cc_task':>14} "
          f"{'kappa_cr_social':>16} {'kappa_cr_task':>14} {'kappa_mean':>11}")
    summary = {}
    for label in labels:
        vals = [results.get(f"{label}/kappa_{d}_{a}", {}).get("kappa", float("nan"))
                for d, a in CELLS]
        mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        summary[label] = {"kappa_cc_social": vals[0], "kappa_cc_task": vals[1],
                          "kappa_cr_social": vals[2], "kappa_cr_task": vals[3],
                          "kappa_mean": mean}
        print(f"{label:<8} {vals[0]:>16.4f} {vals[1]:>14.4f} {vals[2]:>16.4f} "
              f"{vals[3]:>14.4f} {mean:>11.4f}")

    if args.composite:
        sel_path = os.path.join(out_dir, "selection.json")
        with open(sel_path, "w") as f:
            json.dump({k: {"selected": r["selected_model"], "val_kappa": r["kappa"],
                           "selection": r["selection"]}
                       for k, r in results.items() if r.get("selection")}, f, indent=1,
                      default=float)
        print(f"\nselection evidence -> {sel_path}")
        print(f"{'cell':<12}{'chosen':<14}{'cv(xgb)':>10}{'cv(nb)':>10}{'margin':>9}{'val':>10}")
        for k, r in results.items():
            if not r.get("selection"):
                continue
            cv = r["selection"]["cv"]
            cell = k.split("/")[-1].replace("kappa_", "")
            g = lambda m: cv[m]["kappa"] if m in cv else float("nan")
            print(f"{cell:<12}{r['selected_model']:<14}{g('xgb'):>10.4f}"
                  f"{g('gaussian_nb'):>10.4f}{r['selection']['margin_over_default']:>9.4f}"
                  f"{r['kappa']:>10.4f}")

    path = os.path.join(out_dir, "val_kappa.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "feature_counts": counts, "summary": summary,
                   "detail": results}, f, indent=1, default=float)
    print(f"\n-> {path}")

    if wb:
        import wandb
        for fs, s in summary.items():
            for k, v in s.items():
                wb.summary[f"val/{fs}/{k}"] = v
        for key, r in results.items():
            if r.get("selection"):
                cv = r["selection"]["cv"]
                for m, v in cv.items():
                    wb.summary[f"cv/{key}/{m}"] = v["kappa"]
                wb.summary[f"chosen/{key}"] = r["selected_model"]
            if r.get("bags"):
                wb.summary[f"oob/{key}"] = r["oob_kappa"]
            if r.get("kappa_ci95"):
                wb.summary[f"ci_lo/{key}"], wb.summary[f"ci_hi/{key}"] = r["kappa_ci95"]
        wb.log({"val_summary": wandb.Table(
            columns=["model", "n_features_cc", "n_features_cr", "kappa_cc_social",
                     "kappa_cc_task", "kappa_cr_social", "kappa_cr_task", "kappa_mean"],
            data=[[lb, counts[label_fs[lb]]["cc"], counts[label_fs[lb]]["cr"],
                   v["kappa_cc_social"], v["kappa_cc_task"], v["kappa_cr_social"],
                   v["kappa_cr_task"], v["kappa_mean"]] for lb, v in summary.items()])})
        recall_rows, imp_rows = [], []
        for key, r in results.items():
            fs, cell = key.split("/")
            cell = cell.replace("kappa_", "")
            yt, yp = vectors[key]
            wb.log({f"confusion/{fs}/{cell}": wandb.plot.confusion_matrix(
                y_true=yt.tolist(), preds=yp.tolist(), class_names=r["classes"],
                title=f"{fs} {cell} (val, rows = truth)")})
            for c, rc, s, ts, ps in zip(r["classes"], r["recall"], r["support"],
                                        r["true_share"], r["pred_share"]):
                recall_rows.append([fs, cell, c, s, rc, ts, ps])
            for rank, (n, g) in enumerate(r["top_features"], 1):
                imp_rows.append([fs, cell, rank, n, g])
        wb.log({"per_class_recall": wandb.Table(
            columns=["model", "cell", "class", "val_frames", "recall", "true_share",
                     "pred_share"], data=recall_rows)})
        wb.log({"feature_importance": wandb.Table(
            columns=["model", "cell", "rank", "feature", "importance"], data=imp_rows)})
        print(f"[wandb] logged -> {wb.url}")
        wb.finish()


if __name__ == "__main__":
    main()
