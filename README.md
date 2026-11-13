# MERGE — Multimodal Engagement Model

**MERGE** is a multimodal engagement-prediction model for the [MultiMediate'26](https://multimediate-challenge.org/) challenge:

- **Continuous engagement** (CCC) on NoXi, NoXi-J, MPII — [`continuous_model`](continuous_model/README.md):
  multimodal transformer with a per-frame temporal-query regression head
- **Categorical social + task engagement** (Cohen's κ) on PInSoRo —
  [`pinsoro_model`](pinsoro_model/README.md): window statistics + gradient-boosted trees

In `continuous_model` a single forward pass predicts engagement for **everyone in the frame**: a shared
per-node trunk runs once over the flattened `(B·N)` node axis, followed by per-node graph
cross-attention and head decoding.

Note: Works on Linux / Mac only.

## Repository layout

```
continuous_model/       continuous model (NoXi, NoXi-J, MPII)
  models/         network modules (fusion, trunk, graph, domain FiLM, regression head)
  data/           loaders, collation, dataset registry
  training/       trainer, losses, EMA, temporal ensembling, wandb logging
  eval/           validation + submission generation
  metrics/        CCC
  scripts/        train, fit_norm_stats, precompute_features
  configs/        default.yaml
pinsoro_model/    PInSoRo model (feature extraction, XGBoost / naive Bayes, decoding)
```

## Setup

```bash
# GPU training env for continuous_model (PyTorch 2.x + CUDA)
conda env create -f continuous_model/environment_gpu.yml
conda activate gpd-comp

# CPU env for pinsoro_model
conda env create -f environment.yml
```

## Data

Point `data.roots` in [`continuous_model/configs/default.yaml`](continuous_model/configs/default.yaml)
at your local copies (defaults are `/path/to/<dataset>` placeholders), and pass `--root` (or set
`PINSORO_ROOT` in [`pinsoro_model/data.py`](pinsoro_model/data.py)) for PInSoRo.

## Quick start

```bash
# 1. fit normalisation stats (refit whenever the modality set changes)
python continuous_model/scripts/fit_norm_stats.py

# 2. train (2× GPU via torchrun)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    continuous_model/scripts/train.py --config continuous_model/configs/default.yaml

# Single GPU:
CUDA_VISIBLE_DEVICES=0 python continuous_model/scripts/train.py --config continuous_model/configs/default.yaml

# 3. validate
python continuous_model/eval/validate.py \
    --config continuous_model/configs/default.yaml \
    --checkpoint checkpoints/<run_name>/best.pt

# PInSoRo: extract features, then train + score val
python -m pinsoro_model.extract_v2 --root /path/to/pinsoro
python -m pinsoro_model.evaluate --root /path/to/pinsoro --feature-set V2 --composite
```

See [`continuous_model/README.md`](continuous_model/README.md) and [`pinsoro_model/README.md`](pinsoro_model/README.md)
for the full workflows.
