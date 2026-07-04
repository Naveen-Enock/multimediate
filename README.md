# MERGE — Multimodal Engagement Model

**MERGE** is a multimodal engagement-prediction model for the [MultiMediate'26](https://multimediate-challenge.org/) challenge:

- **Continuous engagement** (CCC) on NoXi, NoXi-J, MPII — conditional diffusion-transformer head
- **Categorical social + task engagement** (Cohen's κ) on PInSoRo — per-frame temporal-query head

A single forward pass predicts engagement for **everyone in the frame**: a shared per-node trunk
runs once over the flattened `(B·N)` node axis, followed by per-node graph cross-attention and
head decoding.

Note: Works on Linux / Mac only.

## Repository layout

```
comp_model/
  models/      network modules (fusion, trunk, graph, heads)
  data/        loaders, collation, dataset registry
  training/    trainer, losses, wandb logging
  eval/        validation + submission generation
  metrics/     CCC, kappa
  scripts/     train, fit_norm_stats, precompute_features
  configs/     default.yaml
```

## Setup

```bash
# GPU training env (PyTorch 2.x + CUDA)
conda env create -f comp_model/environment_gpu.yml
conda activate gpd-comp

# (analysis-only CPU env)
conda env create -f environment.yml
```

## Data

Point `data.roots` in [`comp_model/configs/default.yaml`](comp_model/configs/default.yaml)
at your local copies (defaults are `/path/to/<dataset>` placeholders).

## Quick start

```bash
# 1. fit normalisation stats (refit whenever the modality set changes)
python comp_model/scripts/fit_norm_stats.py

# 2. confirm loader + model wiring
pytest comp_model/tests/

# 3. train (2× GPU via torchrun)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    comp_model/scripts/train.py --config comp_model/configs/default.yaml

# 4. validate
python comp_model/eval/validate.py \
    --config comp_model/configs/default.yaml \
    --checkpoint checkpoints/<run_name>/epoch_030.pt
```
