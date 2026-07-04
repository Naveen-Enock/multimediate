# MERGE — Multimodal Engagement Model

MERGE (package `comp_model`) predicts interaction engagement for the MultiMediate'26 challenge:

- **Continuous engagement** (CCC) on NoXi, NoXi-J, MPII — conditional diffusion-transformer head
- **Categorical social + task engagement** (Cohen's κ) on PInSoRo — per-frame temporal-query head

A single forward predicts engagement for every person in the frame.

## Setup

```bash
conda env create -f comp_model/environment_gpu.yml   # PyTorch 2.x + CUDA
conda activate gpd-comp
```

## Data

Obtain the four datasets and point `data.roots` in
[`configs/default.yaml`](configs/default.yaml) at your local copies (defaults are `/path/to/<dataset>`
placeholders). All hyperparameters live in that config; CLI flags override.

## Workflow

```bash
# 1. Fit normalisation stats (refit whenever the modality set changes)
python comp_model/scripts/fit_norm_stats.py --config comp_model/configs/default.yaml

# 2. (optional) Precompute fp16 feature arrays for faster loading
python comp_model/scripts/precompute_features.py --config comp_model/configs/default.yaml

# 4. Train — 2× GPU via torchrun (drop torchrun for single-GPU)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    comp_model/scripts/train.py --config comp_model/configs/default.yaml

# 5. Validate a checkpoint (per-domain CCC + PInSoRo kappa)
python comp_model/eval/validate.py --config comp_model/configs/default.yaml \
    --checkpoint checkpoints/<run>/epoch_030.pt

# 6. Write the challenge submission tree
python comp_model/eval/submit.py --config comp_model/configs/default.yaml \
    --checkpoint checkpoints/<run>/epoch_050.pt --out submission --zip submission.zip
```

### Two-stage training

Omitting `--phase` runs the joint multi-task schedule. The curriculum instead trains in two stages —
phase 1 warms the shared trunk on continuous data, phase 2 fine-tunes the whole model on PInSoRo:

```bash
python comp_model/scripts/train.py --config comp_model/configs/default.yaml --phase 1 --epochs 40
python comp_model/scripts/train.py --config comp_model/configs/default.yaml \
    --phase 2 --init_ckpt checkpoints/<phase1_run>/epoch_040.pt --epochs 15
```


## Repository layout

```
comp_model/
  models/      network modules (fusion, temporal, graph, diffusion / categorical heads)
  data/        loaders, windowing, collation, dataset registry, samplers
  training/    trainer, losses, EMA, temporal ensembling, DDP helpers
  eval/        validate.py (CCC + kappa), submit.py (challenge tree)
  metrics/     CCC, Cohen's kappa
  scripts/     train, fit_norm_stats, precompute_features
  configs/     default.yaml
```
