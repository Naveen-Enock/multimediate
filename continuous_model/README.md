# MERGE — Multimodal Engagement Model

MERGE (package `continuous_model`) predicts **continuous engagement** (CCC) on NoXi, NoXi-J and MPII for
the MultiMediate'26 challenge, with a per-frame temporal-query regression head. A single forward
predicts engagement for every person in the frame.

PInSoRo (categorical social + task engagement) is handled by the separate
[`pinsoro_model`](../pinsoro_model/README.md) package.

## Setup

```bash
conda env create -f continuous_model/environment_gpu.yml   # PyTorch 2.x + CUDA
conda activate gpd-comp
```

## Data

Obtain the datasets and point `data.roots` in [`configs/default.yaml`](configs/default.yaml) at
your local copies (defaults are `/path/to/<dataset>` placeholders). All hyperparameters live in
that config; CLI flags override.

## Workflow

```bash
# 1. Fit normalisation stats (refit whenever the modality set changes)
python continuous_model/scripts/fit_norm_stats.py --config continuous_model/configs/default.yaml

# 2. (optional) Precompute fp16 feature arrays for faster loading
python continuous_model/scripts/precompute_features.py --config continuous_model/configs/default.yaml

# 3. Train — 2× GPU via torchrun
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    continuous_model/scripts/train.py --config continuous_model/configs/default.yaml

# 3b. Single GPU: batch_size.train is the *global* batch, so halve it and set grad_accum 2 to
#     keep the effective batch — and therefore the step count and LR schedule — identical.
CUDA_VISIBLE_DEVICES=0 python continuous_model/scripts/train.py \
    --config continuous_model/configs/default.yaml --batch_size 80 --set train.grad_accum_steps=2

# 4. Validate a checkpoint (per-domain CCC)
#    --eval-stride N (N < window.K) overlaps eval windows so temporal ensembling merges
#    K/N predictions per frame; omit for the single-chunk-per-frame default.
python continuous_model/eval/validate.py --config continuous_model/configs/default.yaml \
    --checkpoint checkpoints/<run>/best.pt --eval-stride 16

# 5. Write the continuous challenge submission tree
python continuous_model/eval/submit.py --config continuous_model/configs/default.yaml \
    --checkpoint checkpoints/<run>/best.pt --out submission --zip submission.zip
```

## Repository layout

```
continuous_model/
  models/      network modules (fusion, temporal, graph, domain FiLM, regression head)
  data/        loaders, windowing, collation, dataset registry, sampler
  training/    trainer, losses, EMA, temporal ensembling, DDP helpers
  eval/        validate.py (CCC), submit.py (challenge tree)
  metrics/     CCC
  scripts/     train, fit_norm_stats, precompute_features
  configs/     default.yaml
```
