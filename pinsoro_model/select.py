"""Pick one model family per cell using grouped cross-validation **inside train**.

Each of the four streams is its own model, so "XGBoost where it wins, naive Bayes where it fixes the
zero" is a per-cell choice. Reading those winners off the val table would be selection on the split
the number is then reported on -- with 12 val children and confidence intervals near +-0.15, that is
not a measurement. So the choice is made here, from training sessions only, and val stays a single
honest pass.

Folds are grouped by session, never finer: windows overlap 5x and the two children of a child-child
session share a room, cameras and a microphone.

* cc has 20 training sessions -> ``GroupKFold(5)``
* cr has 12 and only ~6,400 windows -> leave-one-session-out, because the cr folds differ sharply in
  class mix and that shift is exactly what decides the cr social cell

Out-of-fold predictions are pooled across folds into one kappa, the way a submission is scored,
rather than averaged over per-fold kappas (a fold holding a single-class child has no defined kappa).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from .kappa import cohen_kappa

from . import data as D, models
from .train import fold_posteriors, fold_stats

# Default to XGBoost; a challenger must beat it out-of-fold by at least this to take a cell. With 12
# to 20 sessions the CV estimate is itself noisy, and a margin stops the release flipping families
# on a hundredth.
SWITCH_MARGIN = 0.02
CC_FOLDS = 5


def n_folds_for(n_sessions: int, domain: str) -> int:
    return min(n_sessions, CC_FOLDS) if domain == "cc" else n_sessions


def cv_kappa(train_tabs, domain, axis, feature_set, schema, alpha, seed, params, model):
    """Pooled out-of-fold kappa for one (cell, model), plus per-fold detail."""
    from .evaluate import _decode, fit_cell          # late: evaluate imports this module

    n_classes = D.N_CLASSES[axis]
    cell = np.flatnonzero(train_tabs.mask(domain=domain) & (train_tabs.y[axis] >= 0))
    groups = train_tabs.group[cell]
    sessions = np.unique(groups)
    k = n_folds_for(len(sessions), domain)
    pooled_t, pooled_p, per_fold, covered = [], [], [], 0

    for tr_i, va_i in GroupKFold(n_splits=k).split(cell, groups=groups):
        fit_rows, held = cell[tr_i], cell[va_i]
        assert not (set(train_tabs.group[fit_rows]) & set(train_tabs.group[held])), \
            "CV fold shares sessions between fit and held-out"
        m = fit_cell(train_tabs, domain, axis, feature_set, schema, alpha, seed, params,
                     model=model, rows=fit_rows)
        if m is None:
            continue
        # The held-out sessions are scored with transitions and prior from the fold's own training
        # sessions, so nothing from the held-out child leaks into the decode either.
        trans, prior = fold_stats(train_tabs, fit_rows, axis, n_classes)
        posts = fold_posteriors(m["bst"], train_tabs, held, m["cols"], axis, m["names"])
        if not posts:
            continue
        t, p = _decode(posts, trans, prior, m["stickiness"])
        covered += len(held)
        per_fold.append({"sessions": sorted(set(train_tabs.group[held].tolist())),
                         "frames": int(t.size),
                         "kappa": float(cohen_kappa(t, p, num_classes=n_classes))})
        pooled_t.append(t)
        pooled_p.append(p)

    if not pooled_t:
        return {"kappa": float("nan"), "folds": [], "n_folds": k, "rows_covered": 0}
    kap = cohen_kappa(np.concatenate(pooled_t), np.concatenate(pooled_p), num_classes=n_classes)
    return {"kappa": float(kap), "folds": per_fold, "n_folds": k,
            "rows_covered": int(covered), "rows_total": int(cell.size)}


def choose(train_tabs, domain, axis, feature_set, schema, alpha, seed, params,
           candidates=("xgb", "gaussian_nb"), default="xgb", margin=SWITCH_MARGIN, log=print):
    """Grouped-CV comparison of ``candidates`` for one cell; returns the choice and its evidence."""
    cv = {m: cv_kappa(train_tabs, domain, axis, feature_set, schema, alpha, seed, params, m)
          for m in candidates}
    base = cv[default]["kappa"]
    best, best_k = default, base
    for m, r in cv.items():
        if m == default or not np.isfinite(r["kappa"]):
            continue
        if r["kappa"] > best_k and r["kappa"] - base >= margin:
            best, best_k = m, r["kappa"]
    for m, r in cv.items():
        mark = "  <- chosen" if m == best else ""
        log(f"      cv {m:<14} kappa {r['kappa']:+.4f} over {r['n_folds']} folds "
            f"({r['rows_covered']}/{r.get('rows_total', 0)} rows held out){mark}")
    return {"model": best, "cv": {m: r for m, r in cv.items()},
            "margin_over_default": float(best_k - base), "default": default,
            "switch_margin": margin}
