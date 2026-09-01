# TabML-Guard

**An explainable, drift-aware AutoML pipeline for real-world, time-ordered tabular data.**

TabML-Guard is a configuration-driven machine-learning pipeline. Point it at any
tabular dataset (an OpenML name or a local CSV), tell it the target column and an
optional time column, and it will automatically:

1. detect whether the task is **classification** or **regression**,
2. profile and clean the data and split it **by time** (mimicking real deployment),
3. fit **regularized baselines** (bias–variance foundation),
4. train **boosting ensembles + a stacking meta-model**,
5. compare **hyperparameter-optimization strategies** (Default / Grid / Random / Bayesian) on both accuracy **and** compute cost,
6. explain the best model with **SHAP** and flag **unreliable predictions**,
7. monitor **concept drift** over time and compare a static vs. periodically-retrained model,
8. run **McNemar's significance test**, and
9. write a single self-contained **HTML report** for the end user.

The tool is the deliverable; a dataset (electricity price movement, ELEC2) is just
the demonstration. This is an *implementation* project — the empirical study is its
*output*.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python run.py --config config.yaml
```

The report is written to `results/report.html`. Open it in any browser.

## Point it at a new dataset

Edit **only** `config.yaml`:

```yaml
dataset:
  source: csv
  path: data/my_data.csv
  target: my_target_column
  time_col: my_timestamp_column   # or null if none
task: auto
```

## Project layout

```
tabml-guard/
├── config.yaml        # the only file you edit per dataset
├── requirements.txt
├── run.py             # entry point
├── src/
│   ├── config.py      # load + validate config
│   ├── data.py        # load, profile, auto-detect task, preprocess, temporal split
│   ├── metrics.py     # task-aware metrics
│   ├── baselines.py   # Stage 1
│   ├── boosting.py    # Stage 2
│   ├── tuning.py      # Stage 3
│   ├── explain.py     # Stage 4
│   ├── drift.py       # Stage 5
│   ├── stats.py       # McNemar's + paired tests
│   ├── plots.py       # figure helpers
│   └── report.py      # HTML report assembly
└── results/           # generated figures, tables, best model, report.html
```

## Build status

| Stage | Status |
|-------|--------|
| 0 — Data (load / profile / task-detect / temporal split) | ✅ implemented |
| 1 — Regularized baselines | ✅ implemented |
| 2 — Boosting + stacking | ✅ implemented |
| 3 — HPO comparison | ✅ implemented |
| 4 — SHAP + reliability | ✅ implemented |
| 5 — Drift monitoring | ✅ implemented |
| — McNemar's test | ✅ implemented |
| — HTML report | ✅ implemented (placeholder renders the data profile) |

Running the pipeline today loads the data, profiles it, performs the temporal
split, and writes a placeholder report — so you can confirm the environment and
dataset work on your machine before the analysis stages are filled in.

## Literature backing

The design maps to five reviewed papers: RegStack (ESWA 2025) for regularization +
stacking; Bayesian-Optimized XGBoost (ESWA 2026) for the tuning backbone; SHAP
Error-Compensation (IEEE TAI 2025) for reliability-aware explainability; Concept
Drift with Continuous Kernel Learning (Information Sciences 2024) for drift
monitoring; and Spectral-Clustering Parameter Tuning (IEEE Access 2025) for the
"parameters matter as much as algorithms" theme. See `PROJECT_PLAN.md` for the
full stage-by-stage mapping.
