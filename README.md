# Does the Order of Training Data Influence Optimization?

EPFL — Optimization for Machine Learning, Mini-Project, Spring 2026
Authors: Niels Enkaoua, Rémi Germi, Mohamed Sami Ghrab

This project studies how the order in which training examples are presented
affects the optimization trajectory and generalization of a neural network.
The final report is `final_report.pdf` (source: `final_report.tex`).

## Research question

**How does the training-data order influence optimization speed, gradient
behavior, and test accuracy when the model, optimizer, dataset, and
hyperparameters are kept fixed?**

## Experimental variable

The only experimental variable is the order of the training samples:

1. `random`: standard random reshuffling at every epoch (main baseline).
2. `fixed_random`: one random permutation reused at every epoch.
3. `label_sorted`: all examples sorted by label (strongly non-iid).
4. `label_block_random`: class blocks; class order and within-class order change every epoch.
5. `curriculum_easy`: examples close to their class prototype first.
6. `curriculum_hard`: examples far from their class prototype first.

Two control studies test robustness of the findings:

- **Optimizer control**: AdamW at learning rates 1e-3 and 1e-2 (vs. momentum SGD).
- **Model-class control**: convex logistic regression (vs. the non-convex CNN).

## Fixed components

- Dataset: Fashion-MNIST (stratified 20,000-example training subset)
- Default model: small CNN without BatchNorm or Dropout
- Default optimizer: SGD with momentum 0.9, weight decay 5e-4, batch size 128
- Metrics: train loss, test accuracy, gradient norms, gradient cosine
  similarity, per-class accuracy, forgetting, prediction distributions
- Seeds: 0, 1, 2

## Repository structure

```text
.
├── final_report.tex / final_report.pdf   # the report
├── run.py                                # basic experiment runner
├── run_v3.py                             # extended diagnostics runner
├── build_report_v8_assets.py             # rebuilds all report figures from stored results
├── references.bib
├── requirements.txt
└── src/
    ├── data.py          # dataset loading and order samplers
    ├── models.py        # SmallCNN and LogisticRegression
    ├── train.py         # training loop and metric recording
    ├── plotting.py      # basic plots
    ├── plotting_v3.py   # diagnostic plots (cosine, forgetting, per-class)
    ├── plotting_v8.py   # report figures
    └── utils.py
```

Result directories used by the report (committed so figures can be rebuilt
without retraining): `results_v3_main`, `results_v3_lr0p1`, `results_lr0001`,
`results_adamw_lr0p001`, `results_adamw_lr0p01`, `results_logistic_comparison`.

## Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the report

All numbers and figures in `final_report.pdf` come from the result
directories listed above. To rebuild the figures from the stored results
(no retraining, takes seconds):

```bash
python build_report_v8_assets.py
```

Then compile the report:

```bash
pdflatex final_report.tex
pdflatex final_report.tex
```

To rerun the experiments from scratch:

```bash
# Main experiment and lr=0.1 sweep (results_v3_main, results_v3_lr0p1):
python run_v3.py --epochs 10 --seeds 0 1 2 --max-train-examples 20000

# lr=0.001 rescue runs for the label-based orders (results_lr0001):
python run.py --epochs 10 --seeds 0 1 2 --max-train-examples 20000 \
    --orders label_sorted label_block_random --lr 0.001 --output-dir results_lr0001

# Optimizer and model-class controls (Table II of the report):
python run_v3.py --skip-main --skip-sweep --compare-adamw --compare-logistic
```

Final test accuracies are written to `<results_dir>/tables/final_test_accuracy.csv`.

## Quick smoke test

```bash
python run.py --epochs 1 --seeds 0 --max-train-examples 2000 --orders random fixed_random --no-progress
```

## Key findings

- Random reshuffling is the strongest, most stable baseline (89.5% test
  accuracy); a fixed random order is consistently slightly worse.
- Hard-to-easy curriculum clearly outperforms easy-to-hard.
- Label-sorted and label-block streams collapse to chance accuracy at the
  reference learning rate, producing degenerate single-class predictors and
  non-finite gradient diagnostics; lowering the learning rate softens but
  does not repair the failure.
- AdamW partially rescues class-blocked streams at lr 1e-3 but has an even
  narrower stability region at lr 1e-2.
- A convex logistic-regression model degrades under the same orders but never
  collapses: the catastrophic failure mode requires non-convexity.

See `final_report.pdf` for the full analysis.
