"""Naive Bayes class log-likelihoods as extra XGBoost features (`xgb_nb`).

The nine-model comparison showed XGBoost and Gaussian naive Bayes are complementary **per class** in
every cell -- XGBoost owns the head of the label distribution, NB the tail (cr social `onlooker`:
NB 0.87 recall against XGBoost's 0.00). Instead of combining two models' outputs, NB's log
probabilities become ``n_classes`` extra columns so a single XGBoost learns when to use them, in
combination with the raw features.

**The out-of-fold construction is the whole correctness story.** Features for *training* rows come
from NB models fitted on other sessions via ``GroupKFold``, never on the row's own session. Sessions
are the dependence unit here, and these features identify the recording almost perfectly -- an
in-fold NB would hand XGBoost a session fingerprint that evaporates on unseen
children. Val and test features come from one NB fitted on all of train, which is the standard
stacking arrangement.

Log probabilities rather than probabilities: NB's independence assumption drives its posteriors to
0/1, so the raw scale is saturated and uninformative; the log scale is not.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from . import data as D, linear_models, models

LOGP_FLOOR = 1e-12          # keeps log() finite on NB's saturated posteriors
NB_OOF_FOLDS = 5


def nb_feature_names(axis: str) -> list:
    return [f"nb.logp.{c}" for c in D.CLASS_NAMES[axis]]


def _nb_logp(nb, X, n_classes: int) -> np.ndarray:
    return np.log(np.clip(nb.predict_proba_matrix(X), LOGP_FLOOR, None))


def nb_oof_features(tabs, rows: np.ndarray, cols: np.ndarray, axis: str, alpha: float,
                    nb_params: dict, seed: int, folds: int = NB_OOF_FOLDS):
    """Out-of-fold NB log-probabilities for ``rows``, plus an NB fitted on all of them.

    Returns ``(oof_logp, full_nb)``. Every row's features come from an NB that never saw that row's
    session; ``full_nb`` is for rows outside this set (dev, val, test).
    """
    n_classes = D.N_CLASSES[axis]
    y, pur = tabs.y[axis], tabs.purity[axis]
    groups = tabs.group[rows]
    oof = np.zeros((rows.size, n_classes), np.float64)
    filled = np.zeros(rows.size, bool)

    n_splits = min(folds, len(np.unique(groups)))
    if n_splits >= 2:
        for tr_i, va_i in GroupKFold(n_splits=n_splits).split(rows, groups=groups):
            fit_rows, pred_rows = rows[tr_i], rows[va_i]
            # The assertion that matters: the NB never sees the sessions it is about to score.
            assert not (set(tabs.group[fit_rows]) & set(tabs.group[pred_rows])), \
                "NB out-of-fold split shares sessions between fit and predict"
            w = D.sample_weights(y[fit_rows], pur[fit_rows], n_classes, alpha)
            nb = linear_models.fit(tabs.X[np.ix_(fit_rows, cols)], y[fit_rows], w, n_classes,
                                   "gaussian_nb", nb_params, seed)
            oof[va_i] = _nb_logp(nb, tabs.X[np.ix_(pred_rows, cols)], n_classes)
            filled[va_i] = True
    assert filled.all() or n_splits < 2, "some rows received no out-of-fold NB features"

    w_all = D.sample_weights(y[rows], pur[rows], n_classes, alpha)
    full_nb = linear_models.fit(tabs.X[np.ix_(rows, cols)], y[rows], w_all, n_classes,
                                "gaussian_nb", nb_params, seed)
    if n_splits < 2:                                    # degenerate: fall back to in-fold
        oof = _nb_logp(full_nb, tabs.X[np.ix_(rows, cols)], n_classes)
    return oof, full_nb


class NBAugmentedBooster:
    """An XGBoost booster that appends NB log-probabilities to whatever rows it is given.

    Exposes ``predict_proba_matrix`` and ``coef_importance``, the two methods
    ``models.predict_proba`` / ``models.gain_importance`` dispatch on, so ``fold_posteriors``,
    ``predict_table_frames`` and the wandb importance table need no changes.
    """

    def __init__(self, bst, full_nb, n_classes: int, aug_names: list):
        self.bst, self.full_nb = bst, full_nb
        self.n_classes, self.aug_names = n_classes, aug_names

    def augment(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return np.hstack([X, _nb_logp(self.full_nb, X, self.n_classes)])

    def predict_proba_matrix(self, X) -> np.ndarray:
        return models.predict_proba(self.bst, self.augment(X), self.aug_names)

    def coef_importance(self, feature_names) -> list:
        """Total gain over the augmented names, so the NB columns are visible in the report."""
        return models.gain_importance(self.bst, self.aug_names)
