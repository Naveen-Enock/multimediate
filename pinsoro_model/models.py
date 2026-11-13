"""XGBoost wrapper and the hyperparameter grid.

The low-level ``xgb.train`` API is used rather than ``XGBClassifier`` on purpose: ``num_class`` is
pinned to the axis's full class count, so a fold that happens to contain no ``adultseeking`` window
still produces a probability column for it and every column stays aligned with the global class
index. ``XGBClassifier`` would silently relabel the classes it sees.

Missing values are passed through as ``NaN``. XGBoost learns a default branch direction per split,
which is the right semantics for "the left wrist was not detected in this window" -- imputing would
place those windows in the middle of the feature's real range.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

# Regularisation is budgeted against the number of *independent label runs* (85 for cr social,
# ~1000 for cc), not the number of windows (~48k) -- windows inside one run are near-duplicates.
BASE_PARAMS = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    # xgboost >= 3 probes for a GPU by default; this box's CUDA driver is older than the runtime
    # the wheel was built against, so pin to CPU rather than failing mid-fit.
    "device": "cpu",
    "eta": 0.05,
    "subsample": 0.7,
    "max_depth": 4,
    "min_child_weight": 20.0,
    "colsample_bytree": 0.5,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "nthread": 8,
    "verbosity": 0,
}

# Deliberately small: with tens of sessions, a large search overfits the CV estimate itself.
PARAM_GRID = [
    {"max_depth": 3, "min_child_weight": 40.0, "colsample_bytree": 0.4, "reg_lambda": 10.0},
    {"max_depth": 3, "min_child_weight": 20.0, "colsample_bytree": 0.6, "reg_lambda": 5.0},
    {"max_depth": 4, "min_child_weight": 40.0, "colsample_bytree": 0.4, "reg_lambda": 10.0},
    {"max_depth": 4, "min_child_weight": 20.0, "colsample_bytree": 0.6, "reg_lambda": 5.0},
    {"max_depth": 5, "min_child_weight": 60.0, "colsample_bytree": 0.4, "reg_lambda": 20.0},
    {"max_depth": 6, "min_child_weight": 80.0, "colsample_bytree": 0.3, "reg_lambda": 20.0},
]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]      # class weight = freq^-alpha
MAX_ROUNDS = 600
EARLY_STOPPING = 40


def make_params(n_classes: int, overrides: dict | None = None, seed: int = 0) -> dict:
    p = dict(BASE_PARAMS)
    p.update(overrides or {})
    p["num_class"] = int(n_classes)
    p["seed"] = int(seed)
    return p


def dmatrix(X, y=None, w=None, feature_names=None) -> xgb.DMatrix:
    return xgb.DMatrix(np.asarray(X, np.float32), label=y, weight=w,
                       missing=np.nan, feature_names=feature_names)


def fit(X, y, w, n_classes: int, params: dict, feature_names=None,
        eval_set=None, num_boost_round: int = MAX_ROUNDS, seed: int = 0):
    """Train a booster. ``eval_set = (X, y, w)`` enables grouped early stopping.

    The early-stopping set must come from *held-out sessions of the training folds*, never from the
    outer validation fold -- otherwise the reported CV kappa is optimistic.
    """
    dtrain = dmatrix(X, y, w, feature_names)
    p = make_params(n_classes, params, seed)
    kwargs = {}
    if eval_set is not None:
        Xe, ye, we = eval_set
        kwargs = {"evals": [(dmatrix(Xe, ye, we, feature_names), "es")],
                  "early_stopping_rounds": EARLY_STOPPING, "verbose_eval": False}
    return xgb.train(p, dtrain, num_boost_round=num_boost_round, **kwargs)


def predict_proba(bst, X, feature_names=None) -> np.ndarray:
    # Non-xgboost models (see linear_models.SklearnModel) carry their own probability method, so the
    # whole decode path can stay model-agnostic.
    if hasattr(bst, "predict_proba_matrix"):
        return bst.predict_proba_matrix(X)
    best = getattr(bst, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, bst.num_boosted_rounds())
    return bst.predict(dmatrix(X, feature_names=feature_names), iteration_range=rng)


def gain_importance(bst, feature_names) -> list:
    """``(name, importance)`` pairs, largest first: total gain for trees, |coefficient| for linear."""
    if hasattr(bst, "coef_importance"):
        return bst.coef_importance(feature_names)
    score = bst.get_score(importance_type="total_gain")
    return sorted(((n, float(score.get(n, 0.0))) for n in feature_names),
                  key=lambda kv: -kv[1])
