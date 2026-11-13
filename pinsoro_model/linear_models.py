"""Linear baselines on the V2 window features.

``linreg`` -- literal linear regression: ridge fitted to the **one-hot indicator matrix**, one
column per class, prediction = argmax (Hastie, ESL 4.2). No ordering assumption is needed. Known
caveat: with K >= 3 classes, a class lying between others in feature space can be *masked* and never
win the argmax; the per-class recall table is where that shows up.

``logreg`` -- multinomial (softmax) logistic regression with an L2 penalty.

Both share one pipeline, because neither tolerates what XGBoost did:

* **median imputation + missing indicators.** 47 of 49 V2 features contain NaN (worst 28%), and
  XGBoost's learned per-split default branch has no equivalent here. An indicator column is added
  for every feature missing more than 5% of the time, because the missingness is itself signal -- a
  child leaning over the table loses the keypoint.
* **standardisation.** ``head_dist.mean`` has a median near 751 (mm) while ``r.hand_motion`` sits at
  0.019 (a correlation). Trees are scale-invariant; a penalised linear model is not.

Every statistic is fitted on training rows only, inside an sklearn ``Pipeline``.

The wrapper exposes ``predict_proba_matrix`` and ``coef_importance``, which is all
``models.predict_proba`` / ``models.gain_importance`` need, so the whole decode path -- triangular
overlap averaging, HMM smoothing, per-frame kappa -- is shared with XGBoost unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# A feature missing more than this often gets a companion "was it missing" column.
NAN_INDICATOR_MIN = 0.05
# Winsorising quantiles, learned on train. Needed because the features carry genuine outliers:
# `head_dist` (OpenFace pose_Tz, normally 560-930 mm) reaches 2.5e8 mm on a handful of val windows,
# since the extractor masks by the tracker's success flag but never range-checks the value. Trees
# split on order and are immune; a linear model would let one such window dominate its row.
CLIP_Q = (0.001, 0.999)


class TrainQuantileClip(BaseEstimator, TransformerMixin):
    """Clip every feature to a quantile range measured on the training rows only."""

    def __init__(self, lo=CLIP_Q[0], hi=CLIP_Q[1]):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None, **kw):
        X = np.asarray(X, np.float64)
        self.lo_, self.hi_ = np.nanquantile(X, self.lo, axis=0), np.nanquantile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, np.float64), self.lo_, self.hi_)

# Deliberately small grids: the effective sample size is sessions (20 cc, 12 cr), not windows.
GRID = {
    "linreg": [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)],
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "svm_rbf": [{"C": c} for c in (1.0, 10.0)],
    "svm_linear": [{"C": c} for c in (0.1, 1.0)],
    "lda_shrunk": [{}],                      # Ledoit-Wolf shrinkage is chosen analytically
    "gaussian_nb": [{"var_smoothing": v} for v in (1e-9, 1e-3)],
    "nearest_centroid": [{"shrink_threshold": t} for t in (None, 0.5)],
    "tree_depth3": [{"max_depth": d} for d in (3, 5)],
}
KINDS = tuple(GRID)

# LinearDiscriminantAnalysis and NearestCentroid have no sample_weight in their fit signature, so
# they are fitted unweighted. Reported as-is rather than worked around: class-weighting is part of
# every other model's setup, and hiding the difference would make the comparison misleading.
NO_SAMPLE_WEIGHT = frozenset({"lda_shrunk", "nearest_centroid"})

# SVC is ~O(n^2) in rows. On cc (~26,500 training windows) keep every 5th window, which is exactly
# non-overlapping at a 10 s window and 2 s hop, so almost nothing is lost. cr stays whole -- it has
# only ~6,400 rows, and subsampling would cut `associative` to 2 windows.
SUBSAMPLE_CC = frozenset({"svm_rbf", "svm_linear"})


class SklearnModel:
    """An sklearn pipeline wearing the slice of interface the decode path expects."""

    def __init__(self, pipe, kind, n_classes, ind_cols, n_raw):
        self.pipe, self.kind = pipe, kind
        self.n_classes, self.ind_cols, self.n_raw = n_classes, ind_cols, n_raw

    def _aug(self, X):
        X = np.asarray(X, np.float64)
        if len(self.ind_cols) == 0:
            return X
        return np.hstack([X, np.isnan(X[:, self.ind_cols]).astype(np.float64)])

    def predict_proba_matrix(self, X) -> np.ndarray:
        Xa = self._aug(X)
        est = self.pipe.named_steps["est"]
        if self.kind in ("logreg", "gaussian_nb", "tree_depth3", "lda_shrunk"):
            return self._expand(self.pipe.predict_proba(Xa), est)
        if self.kind in ("svm_rbf", "svm_linear"):
            # Softmax over the one-vs-rest margins rather than Platt scaling: probability=True costs
            # an internal 5-fold refit per candidate, and only the ranking matters for the argmax
            # while the HMM simply needs a distribution.
            d = np.atleast_2d(self.pipe.decision_function(Xa))
            if d.shape[1] == 1:                       # binary fallback
                d = np.hstack([-d, d])
            d = d - d.max(axis=1, keepdims=True)
            e = np.exp(d)
            return self._expand(e / e.sum(axis=1, keepdims=True), est)
        if self.kind == "nearest_centroid":
            # No predict_proba and no decision_function, so score by distance to each centroid in
            # the transformed space.
            Xt = self.pipe[:-1].transform(Xa)
            dist = np.linalg.norm(Xt[:, None, :] - est.centroids_[None, :, :], axis=2)
            e = np.exp(-(dist - dist.min(axis=1, keepdims=True)))
            return self._expand(e / e.sum(axis=1, keepdims=True), est)
        # Ridge indicator scores -> a distribution. Shifting by the row minimum keeps the argmax
        # exactly where the regression put it; only the HMM smoothing sees the scaling.
        s = np.asarray(self.pipe.predict(Xa), np.float64)
        if s.ndim == 1:
            s = s[:, None]
        s = s - s.min(axis=1, keepdims=True) + 1e-6
        return s / s.sum(axis=1, keepdims=True)

    def _expand(self, p, est) -> np.ndarray:
        """Place a model's columns, which cover only the classes it saw, into the full K columns."""
        out = np.zeros((len(p), self.n_classes), np.float64)
        out[:, np.asarray(est.classes_, int)] = p
        return out

    def coef_importance(self, feature_names) -> list:
        """``(name, importance)``, largest first: |coefficient| for linear models, Gini for trees,
        and zeros for models that expose neither (naive Bayes, nearest centroid)."""
        est = self.pipe.named_steps["est"]
        if hasattr(est, "coef_"):
            imp = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
        elif hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, np.float64)
        else:
            imp = np.zeros(len(feature_names) + len(self.ind_cols))
        names = list(feature_names) + [f"{feature_names[c]}.is_missing" for c in self.ind_cols]
        return sorted(((n, float(v)) for n, v in zip(names, imp)), key=lambda kv: -kv[1])


def fit(X, y, w, n_classes: int, kind: str, params: dict, seed: int = 0) -> SklearnModel:
    """Fit one linear baseline. ``X`` may contain NaN; ``w`` are the usual class x purity weights."""
    X = np.asarray(X, np.float64)
    ind_cols = np.flatnonzero(np.isnan(X).mean(axis=0) > NAN_INDICATOR_MIN)
    n_raw = X.shape[1]
    model = SklearnModel(None, kind, n_classes, ind_cols, n_raw)
    Xa = model._aug(X)

    if kind == "linreg":
        est = Ridge(alpha=params["alpha"], fit_intercept=True)
        target = np.zeros((len(y), n_classes), np.float64)   # one-hot; absent classes stay all-zero
        target[np.arange(len(y)), np.asarray(y, int)] = 1.0
    else:
        target = np.asarray(y, int)
        if kind == "logreg":
            est = LogisticRegression(C=params["C"], max_iter=2000, random_state=seed)
        elif kind == "svm_rbf":
            est = SVC(C=params["C"], kernel="rbf", gamma="scale", probability=False,
                      decision_function_shape="ovr", random_state=seed)
        elif kind == "svm_linear":
            est = SVC(C=params["C"], kernel="linear", probability=False,
                      decision_function_shape="ovr", random_state=seed)
        elif kind == "lda_shrunk":
            est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif kind == "gaussian_nb":
            est = GaussianNB(var_smoothing=params["var_smoothing"])
        elif kind == "nearest_centroid":
            est = NearestCentroid(shrink_threshold=params["shrink_threshold"])
        elif kind == "tree_depth3":
            est = DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_leaf=50,
                                         random_state=seed)
        else:
            raise ValueError(f"unknown model {kind!r}")

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clip", TrainQuantileClip()),
                     ("sc", StandardScaler()),
                     ("est", est)])
    if kind in NO_SAMPLE_WEIGHT:
        pipe.fit(Xa, target)
    else:
        pipe.fit(Xa, target, est__sample_weight=np.asarray(w, np.float64))
    model.pipe = pipe
    return model
