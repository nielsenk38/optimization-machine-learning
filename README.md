# Optimization Mini-Project: Does Training-Data Order Matter?

This project studies how the order in which training examples are presented affects the optimization trajectory and generalization of a neural network.

The code is intentionally written in English and organized so that `run.py` reproduces all CSV results and plots used in the report.

## Research question

**How does the training-data order influence optimization speed, gradient behavior, and test accuracy when the model, optimizer, dataset, and hyperparameters are kept fixed?**

## Experimental variable

The only experimental variable is the order of the training samples:

1. `random`: standard random reshuffling at every epoch. This is the main baseline.
2. `fixed_random`: one random permutation reused at every epoch. This controls for the effect of reshuffling.
3. `label_sorted`: all examples are sorted by label. This is a strongly non-iid order.
4. `label_block_random`: class blocks are used, but the class order and within-class order change every epoch.
5. `curriculum_easy`: examples close to their class prototype are seen first.
6. `curriculum_hard`: examples far from their class prototype are seen first.

## Fixed components

- Dataset: Fashion-MNIST
- Default model: small CNN without BatchNorm or Dropout
- Default optimizer: SGD with momentum
- Main metrics: train loss, test accuracy, mean gradient norm
- Repeated seeds: 0, 1, 2 by default

## Folder structure

```text
.
├── run.py
├── requirements.txt
├── README.md
└── src
    ├── data.py
    ├── models.py
    ├── plotting.py
    ├── train.py
    └── utils.py
```

After running experiments, the project creates:

```text
results/
├── raw/                  # one CSV per order and seed
├── figures/              # PNG and PDF plots for the report
└── tables/               # summary CSV files
```

## Setup in VSCode

From the project folder:

```bash
python -m venv .venv
```

Activate the environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have a CUDA GPU, install the PyTorch build recommended by the official PyTorch installation selector, then install the remaining requirements.

## Quick smoke test

Run a small experiment first:

```bash
python run.py --epochs 1 --seeds 0 --max-train-examples 2000 --orders random fixed_random --no-progress
```

Expected output:

- `results/raw/metrics_random_seed0.csv`
- `results/raw/metrics_fixed_random_seed0.csv`
- plots in `results/figures/`

## Main experiment for the report

Recommended command:

```bash
python run.py --epochs 8 --seeds 0 1 2 --max-train-examples 20000
```

For a stronger final run, use the full training set:

```bash
python run.py --epochs 10 --seeds 0 1 2 --max-train-examples 60000
```

If you only want to regenerate figures after training:

```bash
python run.py --plot-only
```

If you want to overwrite previous results:

```bash
python run.py --force
```

## Suggested report structure

### 1. Introduction

Explain why sample order matters for stochastic optimization. State the hypothesis:

> Non-iid orders such as class-sorted batches should make optimization noisier and less stable than random reshuffling, while curriculum orders may change early optimization speed without necessarily improving final test accuracy.

### 2. Method

Describe the dataset, model, optimizer, hyperparameters, order strategies, metrics, and seeds. Mention that all components are fixed except sample order.

### 3. Results

Use these plots:

- `results/figures/train_loss.pdf`
- `results/figures/test_accuracy.pdf`
- `results/figures/grad_norm.pdf`
- `results/figures/final_test_accuracy.pdf`

Suggested comparisons:

- `random` vs `fixed_random`: effect of reshuffling.
- `random` vs `label_sorted`: effect of a pathological non-iid order.
- `curriculum_easy` vs `curriculum_hard`: effect of a simple difficulty-based ordering.

### 4. Discussion

Discuss whether the observed behavior supports the hypothesis. Be explicit about limitations:

- Fashion-MNIST is small and clean.
- The class-prototype curriculum is only a proxy for difficulty.
- Results may differ for larger models or data augmentation.

## Possible extensions

If time allows, add one of the following:

1. Repeat the experiment with `--optimizer adamw --lr 0.001`.
2. Repeat with `--model logistic_regression` to compare a simpler convex-ish baseline.
3. Add a second dataset such as MNIST or CIFAR-10.

## Notes for Codex

Good next prompts for Codex in VSCode:

- "Add a table that reports the mean and standard deviation of final test accuracy for each data order."
- "Add an option to save per-batch loss curves for the first epoch."
- "Check whether all random seeds are correctly controlled."
- "Add a second optimizer experiment with AdamW without duplicating code."
