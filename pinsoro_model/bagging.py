"""Classic bootstrap aggregation over **sessions**.

    draw n sessions with replacement from the n training sessions -> fit -> repeat B times
    -> average the class probabilities

The unit is the session, not the window, which is the standard cluster bootstrap for clustered
data. Windows overlap 5x (neighbours share 8 of their 10 seconds), and resampling windows with
replacement is arithmetically identical to reweighting them -- something ``data.sample_weights``
already does through ``alpha`` -- so it would add no information. In a child-child session the two
children share a room, cameras and a microphone, so the whole session moves together.

A session drawn twice contributes its rows twice, physically duplicated, as the textbook procedure
specifies.

**Out-of-bag does double duty.** The sessions a bag never draws (~37% in expectation) are that
bag's early-stopping set, replacing the single grouped dev slice ``evaluate.fit_cell`` carves out.
Pooling each row's probabilities from only the bags where its session was out-of-bag then gives an
honest train-only estimate, used here to pick the decode stickiness and reported as ``oob_kappa``.

There is no final refit on all of train: in classic bagging the ensemble *is* the model.
"""

from __future__ import annotations

import collections
import re

import numpy as np

from .kappa import cohen_kappa

from . import data as D, models
from .train import STICKINESS_GRID, fold_posteriors, fold_stats


class BaggedBooster:
    """Averages the class probabilities of the per-bag boosters.

    Implements the slice of the xgboost ``Booster`` interface that ``models.predict_proba`` and
    ``models.gain_importance`` use, so every downstream caller -- ``fold_posteriors``,
    ``predict_table_frames``, the importance tables -- works unchanged. Each member is evaluated at
    its own early-stopping iteration, so a requested ``iteration_range`` is ignored.
    """

    def __init__(self, boosters):
        self.boosters = list(boosters)
        self.best_iteration = None

    def __len__(self):
        return len(self.boosters)

    def num_boosted_rounds(self):
        return max(b.num_boosted_rounds() for b in self.boosters)

    def predict(self, dmat, iteration_range=None, **kw):
        acc = None
        for b in self.boosters:
            best = getattr(b, "best_iteration", None)
            rng = (0, best + 1) if best is not None else (0, b.num_boosted_rounds())
            p = b.predict(dmat, iteration_range=rng)
            acc = p if acc is None else acc + p
        return acc / len(self.boosters)

    def get_score(self, importance_type="total_gain"):
        tot = collections.defaultdict(float)
        for b in self.boosters:
            for k, v in b.get_score(importance_type=importance_type).items():
                tot[k] += v / len(self.boosters)
        return dict(tot)


def draw_sessions(sessions: np.ndarray, rng) -> tuple:
    """One classic bootstrap draw: ``(counts, out-of-bag sessions)``."""
    drawn = rng.choice(sessions, size=len(sessions), replace=True)
    counts = collections.Counter(drawn.tolist())
    oob = np.array([s for s in sessions if s not in counts])
    return counts, oob


def bag_rows(rows: np.ndarray, groups: np.ndarray, counts) -> np.ndarray:
    """Row indices of one bag; a session drawn ``c`` times appears ``c`` times."""
    parts = []
    for s, c in counts.items():
        idx = rows[groups == s]
        parts.extend([idx] * c)
    return np.concatenate(parts) if parts else np.empty(0, np.int64)


def _decode_posts(posts, trans, prior, stickiness, n_classes):
    from .evaluate import _kappa      # late: evaluate imports this module
    return _kappa(posts, trans, prior, stickiness, n_classes)


def fit_bagged_cell(train_tabs, domain, axis, feature_set, schema, alpha, seed, params, bags,
                    drop=None):
    """Fit ``bags`` bootstrap members for one (domain, axis) cell. Same return shape as
    ``evaluate.fit_cell``, plus ``oob_kappa`` / ``bags`` / ``rounds_per_bag``."""
    n_classes = D.N_CLASSES[axis]
    cols = D.columns_for(train_tabs.feat_names, feature_set, domain, schema)
    if drop:
        pat = re.compile(drop)
        cols = np.array([i for i in cols if not pat.search(train_tabs.feat_names[i])], np.int64)
    names = [train_tabs.feat_names[i] for i in cols]

    tr_all = np.flatnonzero(train_tabs.mask(domain=domain) & (train_tabs.y[axis] >= 0))
    if tr_all.size == 0:
        return None
    groups = train_tabs.group[tr_all]
    sessions = np.unique(groups)
    y, pur = train_tabs.y[axis], train_tabs.purity[axis]

    rng = np.random.default_rng(seed)
    boosters, rounds, oob_sizes = [], [], []
    acc = np.zeros((tr_all.size, n_classes), np.float64)
    cnt = np.zeros(tr_all.size, np.int64)

    for b in range(bags):
        for _ in range(10):                       # redraw if every session landed in the bag
            counts, oob = draw_sessions(sessions, rng)
            if oob.size:
                break
        if not oob.size:
            continue
        in_rows = bag_rows(tr_all, groups, counts)
        oob_rows = tr_all[np.isin(groups, oob)]
        w_in = D.sample_weights(y[in_rows], pur[in_rows], n_classes, alpha)
        w_oob = D.sample_weights(y[oob_rows], pur[oob_rows], n_classes, alpha)
        bst = models.fit(train_tabs.X[np.ix_(in_rows, cols)], y[in_rows], w_in, n_classes, params,
                         feature_names=names,
                         eval_set=(train_tabs.X[np.ix_(oob_rows, cols)], y[oob_rows], w_oob),
                         seed=seed + b)
        boosters.append(bst)
        rounds.append(int(getattr(bst, "best_iteration", bst.num_boosted_rounds() - 1)) + 1)
        oob_sizes.append(int(oob.size))
        pos = np.searchsorted(tr_all, oob_rows)
        acc[pos] += models.predict_proba(bst, train_tabs.X[np.ix_(oob_rows, cols)], names)
        cnt[pos] += 1

    if not boosters:
        return None

    # ── out-of-bag selection of the decode setting ──────────────────────────
    trans, prior = fold_stats(train_tabs, tr_all, axis, n_classes)
    have = cnt > 0
    oob_grid, oob_kappa, stickiness = {}, float("nan"), 0.0
    if have.any():
        posts = fold_posteriors(None, train_tabs, tr_all[have], cols, axis, names,
                                proba=acc[have] / cnt[have][:, None])
        oob_grid = {float(s): _decode_posts(posts, trans, prior, s, n_classes)
                    for s in STICKINESS_GRID}
        stickiness = max(oob_grid, key=lambda s: oob_grid[s] if np.isfinite(oob_grid[s]) else -9)
        oob_kappa = oob_grid[stickiness]

    return {"bst": BaggedBooster(boosters), "cols": cols, "names": names,
            "trans": trans, "prior": prior, "stickiness": float(stickiness),
            "rounds": int(np.median(rounds)), "rounds_per_bag": rounds,
            "dev_grid": oob_grid, "oob_grid": oob_grid, "oob_kappa": float(oob_kappa),
            "bags": len(boosters), "mean_oob_sessions": float(np.mean(oob_sizes)),
            "n_sessions": int(sessions.size),
            "n_train_windows": int(tr_all.size),
            "n_train_sessions": int(len(np.unique(groups)))}


def bag_curve(m, tabs, rows, axis, n_classes, points=(1, 2, 5, 10, 20)):
    """Kappa as a function of ensemble size, to show whether the ensemble has converged."""
    from .evaluate import _kappa
    out = {}
    for b in points:
        if b > len(m["bst"]):
            continue
        sub = BaggedBooster(m["bst"].boosters[:b])
        posts = fold_posteriors(sub, tabs, rows, m["cols"], axis, m["names"])
        out[str(b)] = float(_kappa(posts, m["trans"], m["prior"], m["stickiness"], n_classes))
    return out
