# HiRes vs LoRes Vessel Segmentation Study

Cross-validation experiment comparing nnU-Net models trained on high-resolution
and low-resolution arterial CT scans, with and without clDice in the loss function.

**2×2 experiment matrix:**

| Dataset | Loss |
|---------|------|
| HiRes   | `dice_ce` |
| HiRes   | `cldice`  |
| LoRes   | `dice_ce` |
| LoRes   | `cldice`  |

Results are aggregated into `results/comparison.md` (mean ± std Dice and haus95
across all cross-validation cases).

## Prerequisites

- MIST installed with training dependencies (`pip install "mist-medical[train]"`)
- Access to `/rsrch3/ip/dtfuentes/github/oncopigdata/` (the source data)
- NVIDIA GPU(s)
- `tabulate` Python package for Markdown table output (`pip install tabulate`)

## Quickstart

```bash
git clone <this-repo>
cd hireslores

# Run the full pipeline
make all
```

## Step-by-step

### 1. Setup — create symlink directory structure

```bash
make setup
```

Creates `datasets/hires/raw/train/` and `datasets/lores/raw/train/` with
per-patient subdirectories containing symlinks to the original NIfTI files.
No data is copied. Run this on the cluster where `/rsrch3/...` is mounted.

### 2. Analyze — generate MIST configuration

```bash
make analyze
```

Runs `mist_analyze` for each dataset. Outputs `config.json`, `train_paths.csv`,
and `fg_bboxes.csv` into `results/hires_base/` and `results/lores_base/`.

### 3. Preprocess — convert NIfTI to NumPy

```bash
make preprocess
```

Runs `mist_preprocess` for each dataset. The resulting numpy arrays in
`numpy/hires/` and `numpy/lores/` are shared by both loss-function experiments
on the same dataset.

### 4. Train — run all four experiments

```bash
make train
```

Runs `mist_train` for all four `(dataset, loss)` combinations. Each experiment
writes to its own results directory:

```
results/
├── hires_dice_ce/
├── hires_cldice/
├── lores_dice_ce/
└── lores_cldice/
```

To run experiments in parallel (if you have multiple GPUs or a multi-GPU node):

```bash
make -j4 train
```

### 5. Summary — collect results

```bash
make summary
```

Reads `results.csv` from each experiment and writes a Markdown comparison table
to `results/comparison.md`.

## Tunable parameters

All parameters can be overridden on the command line:

```bash
make all NFOLDS=3 EPOCHS=500 MODEL=mednext-base BATCH=4 WORKERS=8
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NFOLDS`   | `5`       | Number of cross-validation folds |
| `EPOCHS`   | `300`     | Training epochs per fold |
| `MODEL`    | `nnunet`  | Architecture (`nnunet`, `mednext-base`, etc.) |
| `BATCH`    | `2`       | Batch size per GPU |
| `WORKERS`  | `4`       | Parallel workers for analyze/preprocess/evaluate |

## Cleaning up

```bash
make clean       # remove results/ and numpy/  (keeps symlinks)
make distclean   # also remove datasets/*/raw/ symlink trees
```

## Data

`data/hires.csv` and `data/lores.csv` are pre-split from `vesseltraining.csv`
(12 cases each, 6 subjects × 2 timepoints). Source data lives on the cluster at
`/rsrch3/ip/dtfuentes/github/oncopigdata/`.
