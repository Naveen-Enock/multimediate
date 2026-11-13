# `pinsoro_model` — window statistics + gradient-boosted trees for PInSoRo

Predicts PInSoRo **social** and **task** engagement (per-frame, unweighted Cohen's κ). Fixed-length
statistics are computed over overlapping windows of the raw SSI streams, cached as one `.npz` per
(session, target child) **inside the session directory**, and fed to XGBoost (with Gaussian naive
Bayes as a per-cell alternative). The package is self-contained (`sessions.py` holds the PInSoRo
on-disk layout, `streams.py` the SSI reader, `kappa.py` the metric); run it from the repo root.

## Setup

```bash
conda env create -f environment.yml    # CPU env: numpy, scipy, scikit-learn, xgboost
conda activate gpd
```

Point `--root` (default `PINSORO_ROOT` in `data.py`) at your local PInSoRo copy.

## Pipeline

```bash
python -m pinsoro_model.columns --root /path/to/pinsoro   # check the stream column layouts
python -m pinsoro_model.extract_v2 --jobs 8               # -> <session>/<role>.v2_feats.npz

# train on train, choose one model family per cell by grouped CV inside train, score val once
python -m pinsoro_model.evaluate --feature-set V2 --composite --run-name v5-composite
```

The tiered (T0/T1/T2) feature tables and the grouped-CV + prediction path:

```bash
python -m pinsoro_model.extract --overwrite               # -> <session>/<role>.simple_feats.npz
python -m pinsoro_model.baselines                         # majority + session-mode oracle
python -m pinsoro_model.train --run-name simple01         # grouped CV + final fit
python -m pinsoro_model.predict --run-name simple01 --tier T1 --splits test
```

Artifacts land under `pinsoro_model/runs/<run-name>/` (gitignored). `predict` writes the challenge
layout `runs/<run>/submission/pinsoro-<domain>/<session>/<role>.<axis>_engagement.prediction.csv`,
one class string per 30 fps frame. Pass `--no-wandb` to skip W&B logging.

## Model

One model per **cell** = (domain, axis): cc (child-child) and cr (child-robot) are fitted
separately, since their label supports barely overlap and the partner is a child in one and a robot
in the other. XGBoost uses the low-level `xgb.train` API (`multi:softprob`) so `num_class` stays
pinned to the axis's full class count; `NaN` is passed through rather than imputed.

* **Composite** (`select.py`): per cell, grouped CV inside train picks between XGBoost and Gaussian
  naive Bayes (`linear_models.py`); val is scored once with the chosen family.
* **Grouping by session.** Windows overlap 5× and the two children of a cc session share a room,
  cameras and a microphone, so every split is grouped by session.
* **Early stopping and decode setting** are chosen on a grouped slice of train, never on val.
* Optional session-level bagging (`bagging.py`) and NB-log-likelihood stacking (`stacking.py`).

## Decoding

Window probabilities are accumulated onto the 30 fps grid with a triangular weight peaked at each
window's centre, then optionally Viterbi-smoothed (`decode.py`). `stickiness` tempers the
transition matrix: the empirical 30 fps self-transition probability is ~0.9995, so an untempered
chain collapses each session to one label. `stickiness = 0` is plain argmax, reported as
`kappa_raw`.

## Windows

Windows are enumerated on the native 30 fps annotation grid: **10 s wide, 2 s hop**. Every other
stream is sliced by the matching time interval rather than resampled. A window's label is the
**mode** of its valid frame labels and `purity` (the modal share) is the training sample weight.
Nested 30 s and 60 s windows sharing each centre supply long-range context (T2).

## Features

**V2** (`feature_sets.py`, the single source of truth) is a curated set — 49 features for cc, 39 for
cr — covering attention (gaze), head orientation, hand activity, affect (AUs), observability
(tracking rates), dyadic coupling (cc only) and context. Angles are role-aligned for the mirrored
seats; no age, gender, session position, room audio or env camera.

**T0 / T1 / T2** are column tiers of one 442-feature table (`feature_schema.json`):

| tier | contents |
|---|---|
| **T0** | room speech activity, gaze, head pose, wrist motion, dyadic synchrony, metadata |
| **T1** | T0 plus the full per-child OpenFace2 / OpenPose blocks and all dyadic contrasts |
| **T2** | T1 plus the env node, the opaque audio channels, and the 30 s / 60 s context blocks |

Deep embeddings (Swin/CLIP/DINO/VideoMAE/w2v-BERT2/XLM-R) are excluded: with 20 cc / 12 cr training sessions they
act as session fingerprints that boosted trees memorise.

## Cache schema

`<session_dir>/<role>.{simple_feats,v2_feats}.npz`:

| key | shape / type | meaning |
|---|---|---|
| `X` | `(n_win, D)` float32 | feature matrix; `NaN` = not computable |
| `feat_names` | `(D,)` str | ordered feature names |
| `schema_hash` | str | md5 of `feat_names`; asserted identical across every table |
| `y_social`, `y_task` | `(n_win,)` int8 | modal class, `-1` where unannotated (all of `test-*`) |
| `purity_social`, `purity_task` | `(n_win,)` float32 | training sample weight |
| `frame_start`, `frame_end` | `(n_win,)` int32 | native 30 fps indices, for per-frame reassembly |
| `t_center` | `(n_win,)` float32 | seconds |
| `n_frames_30fps` | int | session length |
| `session`, `role`, `partner`, `domain`, `split`, `partner_is_robot` | str / bool | metadata |
| `social_runs`, `task_runs` | int | independent label runs — the effective sample size |

## Data caveats

* **The audio is a shared room microphone**, not a per-child signal: the two children's
  `audio.egemapsv2` streams are 33–58% bit-identical. Acoustic features are emitted as `room.*`
  and no turn-taking feature is computed; all dyadic features are visual.
* **`audio.egemapsv2` is not the canonical eGeMAPSv02 vector**: its 88 columns are 44 interleaved
  pairs (even columns `{-2, 0, +2}`, odd columns continuous), so channels are treated as opaque.
* **OpenPose col 54 is a detection score**, not a torso scale; torso scale comes from shoulder width.
* Scored roles follow `registry.target_roles`: children only, so a cc session yields two tables and
  a cr session one (the robot is a partner, never a target).
