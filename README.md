# TabML-Guard

[![PyPI](https://img.shields.io/pypi/v/tabml-guard?color=blue)](https://pypi.org/project/tabml-guard/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://pypi.org/project/tabml-guard/)
[![License](https://img.shields.io/badge/License-MIT-0f766e)](LICENSE)
[![CI](https://github.com/Ayush-Gole8/tabml-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayush-Gole8/tabml-guard/actions)
[![Config-driven](https://img.shields.io/badge/config--driven-yes-14213d)](config.yaml)

**An explainable, drift-aware AutoML pipeline and diagnostic framework for real-world, time-ordered tabular data.**

Point TabML-Guard at any tabular dataset (an OpenML name/id, local CSV, or in-memory DataFrame), name the target column and an optional time column, and it runs the entire modern supervised-learning workflow end to end: auto-detects classification vs. regression, splits the data *by time*, fits regularized baselines, trains boosting ensembles and a stacking meta-model, compares four hyperparameter-optimization strategies on **accuracy *and* compute cost**, explains the winning model with SHAP, flags its least-reliable predictions, monitors concept drift over time, runs statistical significance tests, and writes self-contained HTML and PDF diagnostic reports with a plain-English verdict on top.

The **tool is the deliverable**; the datasets are just demonstrations. Nothing in `tabml_guard/` knows about any specific dataset — every dataset detail lives in the config or CLI arguments.

![TabML-Guard example outputs](https://raw.githubusercontent.com/Ayush-Gole8/tabml/main/assets/sample_outputs.png)

---

## Table of contents

- [Why TabML-Guard](#why-tabml-guard)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The pipeline](#the-pipeline)
- [Configuration reference](#configuration-reference)
- [Run it on your own data](#run-it-on-your-own-data)
- [Outputs](#outputs)
- [Honest evaluation: temporal split & drift](#honest-evaluation-temporal-split--drift)
- [Extending TabML-Guard](#extending-tabml-guard)
- [Project structure](#project-structure)
- [Engineering notes](#engineering-notes)
- [Documentation](#documentation)
- [Literature backing](#literature-backing)
- [Roadmap & limitations](#roadmap--limitations)
- [License](#license)

---

## Why TabML-Guard

Most tabular ML code is written once, for one dataset, and quietly bakes in assumptions that make its numbers optimistic - it shuffles time-ordered rows, tunes on the test set, or reports a single accuracy figure with no notion of whether the model will still work next month. TabML-Guard is the opposite: a **reusable system** that makes the honest, defensible version of that workflow the default.

- **One file to touch.** Every dataset-specific decision lives in `config.yaml`. The pipeline code is generic.
- **Leakage-safe by construction.** The preprocessor is re-fit inside every cross-validation fold, and the temporal test set is never seen during model selection or tuning.
- **Cost-aware tuning.** It doesn't just ask "does Bayesian optimization win?" - it reports the wall-clock and evaluation budget each strategy spent, so you can decide whether tuning is worth it.
- **Trust, not just scores.** SHAP explains *why*, a reliability layer flags *which predictions to double-check*, and drift monitoring answers *will this keep working?*
- **Graceful degradation.** Optional dependencies (CatBoost, Optuna, `river`, SHAP) are lazily imported; if one is missing, that step is skipped with a note instead of crashing the run.
- **A report you can hand to a non-expert.** The output is a single self-contained `.html` file - no server, no notebook, no external assets.

---

## Installation

Requires **Python 3.10–3.12**.

### From PyPI
```bash
pip install tabml-guard
```

### From Source (Development)
```bash
git clone https://github.com/Ayush-Gole8/tabml-guard.git
cd tabml-guard
python -m venv .venv
# Activate:
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux

pip install -e ".[all]"
```

---

## Quickstart

### 1. Command Line Interface (CLI)

Run TabML-Guard directly on any CSV file:
```bash
tabml-guard run --data sales.csv --target revenue --time timestamp
```

Or run using a declarative YAML configuration file (e.g. on the built-in OpenML electricity benchmark):
```bash
tabml-guard run --config config.yaml
```

Scaffold a fresh configuration template in your working directory:
```bash
tabml-guard init
```

### 2. Python Programmatic API

Use TabML-Guard inside your Python scripts or Jupyter Notebooks:
```python
import pandas as pd
from tabml_guard import TabMLGuard

df = pd.read_csv("my_data.csv")

# Initialize and fit
guard = TabMLGuard(
    target="churn",
    time_col="signup_date",  # or None for standard cross-validation
    task="auto",             # auto-detects classification vs. regression
    output_dir="audit_results",
)
results = guard.fit(df)

# Export interactive HTML and formal PDF reports
guard.save_report("churn_audit.html", fmt="html")
guard.save_report("churn_audit.pdf", fmt="pdf")

# Production-ready scikit-learn pipeline (preprocessor + champion model)
model = guard.get_best_model()
predictions = guard.predict(df.drop(columns=["churn", "signup_date"]))
```

When execution completes, inspect the self-contained reports:
```
results/electricity/report_electricity.html
results/electricity/report_electricity.pdf
```

---

## The pipeline

TabML-Guard executes six coordinated stages (S0–S5) end to end, threading intermediate state in memory and producing a comprehensive diagnostic verdict:

```text
[ Raw Data / CSV / OpenML / DataFrame ]
                   │
                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 0 · Data Ingestion, Profiling & Temporal Splitting               │
│ • Auto-detects task (Classification vs. Regression) from target column  │
│ • Builds leakage-safe preprocessor (fit strictly on training folds)    │
│ • Preserves chronological order (earliest rows train, latest test)     │
├────────────────────────────────────────────────────────────────────────┤
│ Stage 1 · Regularized Linear Baselines                                 │
│ • Fits L1 (Lasso) sparsity, L2 (Ridge) shrinkage, and ElasticNet       │
│ • Traces full regularization path to expose bias-variance sweet spots  │
├────────────────────────────────────────────────────────────────────────┤
│ Stage 2 · Modern Ensembles & Stacking Meta-Model                       │
│ • Trains XGBoost, LightGBM, CatBoost, and Random Forest                │
│ • Fits regularized stacking meta-model (RegStack architecture)         │
├────────────────────────────────────────────────────────────────────────┤
│ Stage 3 · Cost-Accounted Hyperparameter Optimization (HPO)             │
│ • Compares Default vs. Grid vs. Random vs. Bayesian (Optuna TPE)       │
│ • Strictly matched evaluation budgets: evaluates Score vs. Compute     │
├────────────────────────────────────────────────────────────────────────┤
│ Stage 4 · Explainability & Prediction Reliability Triage               │
│ • Model-aware SHAP attributions (global importances + local drivers)   │
│ • Calibrated confidence guardrail: routes uncertain cases to human     │
├────────────────────────────────────────────────────────────────────────┤
│ Stage 5 · Concept Drift & Statistical Significance                     │
│ • Prequential walk-forward evaluation: races frozen vs. retrained model│
│ • Population Stability Index (PSI) tracks covariate shift over time    │
│ • McNemar's (classification) / Wilcoxon (regression) significance tests│
└────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
[ Self-Contained Interactive HTML Report + Publication-Ready PDF Report ]
```

| Stage | What it achieves |
|---|---|
| **S0 · Ingestion & Split** | Ingests from CSV, OpenML, or DataFrame, profiles schema and missingness, infers task type, builds a leakage-safe preprocessor, and splits chronologically so the model never peeks into the future. |
| **S1 · Baselines** | Establishes the honest yardstick with regularized linear models (L1/L2/ElasticNet) and traces a regularization path to reveal whether regularization strength actually moves the fit. |
| **S2 · Ensembles** | Trains XGBoost, LightGBM, CatBoost, Random Forest, plus a stacking ensemble with a regularized meta-learner, and selects the strongest booster for optimization. |
| **S3 · Cost-Accounted HPO** | Runs Default vs. Grid vs. Random vs. Bayesian (Optuna) at an **equal evaluation budget**, tracking score alongside wall-clock compute to measure if tuning justified its cost. |
| **S4 · Explainability & Triage** | Computes SHAP feature attributions, then evaluates a **reliability layer** that flags low-confidence predictions, measuring how many real-world errors are caught by human review. |
| **S5 · Drift & Significance** | Replays the held-out test stream in ordered chunks, compares frozen vs. sliding-window retraining, tracks feature PSI, and runs paired statistical tests (McNemar / Wilcoxon). |
| **Deliverables** | Persists the champion scikit-learn pipeline (`best_model.pkl`) and compiles standalone, portable HTML and PDF diagnostic reports. |

---

## Configuration reference

The config is a single YAML file parsed into typed dataclasses (`src/config.py`), validated up front with actionable error messages. Every key under `dataset:` maps one-to-one to a loader field, so a typo raises immediately rather than being silently ignored.

### `dataset`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source` | `openml` \| `csv` | `openml` | Where the data comes from. |
| `name` | string | `null` | OpenML dataset name (e.g. `electricity`). Uses `version`. |
| `data_id` | int | `null` | OpenML numeric id (canonical). **Overrides `name`** if both are set. |
| `path` | string | `null` | CSV path relative to the project root (required when `source: csv`). |
| `version` | int | `1` | OpenML version (ignored when `data_id` is set). |
| `target` | string | - | **Required.** The column to predict. |
| `time_col` | string | `null` | Column giving chronological order. `null` → random split + drift skipped. |
| `use_time_as_feature` | bool | `false` | Keep the time column as a model input (normally used only for ordering). |
| `drop_cols` | list | `[]` | Columns to exclude: IDs, free text, and above all **leakage**. |

### Other sections

| Section · key | Default | Meaning |
|---|---|---|
| `task` | `auto` | `auto` \| `classification` \| `regression`. `auto` detects from the target. |
| `split.train_frac` / `split.test_frac` | `0.6` / `0.4` | Temporal split fractions (earliest → train, latest → test; no shuffle). |
| `models.boosters` | `[xgboost, lightgbm, catboost, random_forest]` | Which ensembles to train. |
| `models.stacking` | `true` | Add a stacking meta-model. |
| `tuning.budget_trials` | `40` | Optuna trials; Random search matched to the same budget. |
| `tuning.cv_splits` | `3` | Number of `TimeSeriesSplit` folds during tuning. |
| `tuning.scoring` | `null` | `null` → auto (ROC-AUC for classification, negative RMSE for regression). |
| `drift.n_chunks` | `10` | How many time-ordered chunks the test stream is cut into. |
| `drift.retrain` | `sliding` | `none` \| `periodic` \| `sliding` - how the adaptive model updates. |
| `report.title` | `"TabML-Guard Report"` | Title shown at the top of the report. |
| `report.fbeta` | `2.0` | β for the F-β score (2.0 weights recall higher). |
| `random_state` | `42` | Global seed for reproducibility. |
| `output_dir` | `results` | Where figures, tables, the model and the report are written. |

---

## Run it on your own data

Copy the closest shipped config, change `path` / `target` / `time_col` (and `drop_cols` if there's any leakage), and run. Nothing else changes:

```yaml
dataset:
  source: csv
  path: data/my_data.csv
  target: my_target_column
  time_col: my_timestamp_column   # or null if the data has no time order
  drop_cols: [row_id]             # exclude IDs / leakage
task: auto
```

```bash
python run.py --config configs/my_data.yaml
```

The repo ships **five ready-to-run configs** that between them exercise every branch of the tool - both task types, the significance-test swap, the classification-only reliability layer, categorical one-hot encoding, and leakage removal:

| Config | Task | Demonstrates |
|---|---|---|
| `config.yaml` (electricity) | Classification | The baseline case; runs from OpenML with no download. |
| `configs/occupancy_detection.yaml` | Classification | A clean, all-numeric second classification story. |
| `configs/metro_traffic.yaml` | Regression | Flips the whole pipeline to RMSE/MAE/R² + Wilcoxon. |
| `configs/appliances_energy.yaml` | Regression | Dense numeric data - best case for PSI drift + the regularization path. |
| `configs/bike_sharing.yaml` | Regression | The `drop_cols` leakage demo. |

> See **[DATASETS.md](DATASETS.md)** for the full guide: where to download each CSV, what insight each run should surface, and the two built-in safety nets (a wrong column name prints the available columns; a high-cardinality categorical warns before it blows up memory).

---

## Outputs

Every run writes into `output_dir` (default `results/`):

```
results/
├── report_<dataset>.html    # ← the headline deliverable: open this
├── best_model.pkl           # the best fitted pipeline (joblib) - ready to reuse
├── tables/
│   ├── profile.json         # dataset profile (shape, dtypes, target stats, time span)
│   └── results.json         # JSON-safe per-stage summaries (the report's data)
└── figures/
    ├── shap_importance.png
    ├── reliability.png      # classification runs only
    ├── drift_performance.png
    └── drift_psi.png
```

The report embeds every figure as base64, so a single `.html` file is fully portable - you can email it or drop it on a static host and it just works. The figures above (shown in the banner) are the actual PNGs a run produces.

---

## Honest evaluation: temporal split & drift

The feature that most separates TabML-Guard from a typical notebook is that it never pretends the future is available at training time.

```text
[ Training Window: Earliest 60% ] ──> [ Held-Out Test Stream: Latest 40% ]
                                                 │
                                                 ├──> Chunk 1 (Evaluate ──> Retrain)
                                                 ├──> Chunk 2 (Evaluate ──> Retrain)
                                                 └──> Chunk N (Evaluate ──> Retrain)
```

When a `time_col` is set, rows are sorted chronologically and split by time - the model trains on the past and is tested on the future. The test stream is then cut into `n_chunks` ordered slices, and two models are compared across them:

- a **frozen** model (trained once, never updated), and
- an **adaptive** model that, for each chunk, *predicts first and only then retrains* on it (`none` / `periodic` / `sliding` window) - so there is never any look-ahead.

Alongside performance, a **Population Stability Index (PSI)** trajectory tracks whether the *inputs themselves* are shifting, and McNemar/Wilcoxon tests confirm whether the differences are statistically real rather than noise. The gap between the frozen and adaptive curves is a direct, visual answer to "will this model keep working, and do I need to retrain it?"

![Static vs adaptive performance over time](https://raw.githubusercontent.com/Ayush-Gole8/tabml/main/assets/drift_performance.png)

---

## Extending TabML-Guard

Because every stage funnels through one shared `fit_evaluate` path and a typed config, extensions are localized:

- **Add a model / booster** → add a `_build_*` constructor in `tabml_guard/boosting.py`, register it, and (optionally) add its search space to `tabml_guard/tuning.py` so it can be tuned.
- **Add a metric** → extend `classification_metrics` / `regression_metrics` in `tabml_guard/metrics.py`; every stage picks it up automatically.
- **Add a tuning strategy** → add a `_run_*` function in `tabml_guard/tuning.py` following the existing `refit=False` → pick-from-CV → single refit contract, so its cost is measured the same way.
- **Add a drift detector** → extend `_detect_changes` in `tabml_guard/drift.py` (the `river` ADWIN hook shows the pattern; it degrades gracefully when the library is absent).
- **Restyle the report** → it's pure-stdlib string assembly in `tabml_guard/report.py`; no template engine to learn.

---

## Project structure

```
tabml-guard/
├── pyproject.toml         # PEP 517/518 build specification & dependencies
├── CHANGELOG.md           # version history & release notes
├── LICENSE                # MIT license
├── config.yaml            # reference configuration (electricity benchmark)
├── configs/               # ready-to-run configurations for multiple tabular regimes
├── tabml_guard/           # THE official installable Python package
│   ├── __init__.py        # public API (TabMLGuard, load_config, Config)
│   ├── __version__.py     # single source of version truth (0.1.0)
│   ├── py.typed           # PEP 561 typing marker for IDEs & mypy
│   ├── cli.py             # CLI entry point (tabml-guard run/init/index/version)
│   ├── pipeline.py        # TabMLGuard class (high-level Python API)
│   ├── config.py          # load + validate config into typed dataclasses
│   ├── data.py            # S0: load, profile, task-detect, preprocess, temporal split
│   ├── modeling.py        # shared fit/evaluate core (single source of truth)
│   ├── metrics.py         # task-aware metrics
│   ├── baselines.py       # S1: regularized baselines + regularization path
│   ├── boosting.py        # S2: boosting ensembles + stacking
│   ├── tuning.py          # S3: Default / Grid / Random / Bayesian + cost accounting
│   ├── explain.py         # S4: SHAP + reliability triage
│   ├── drift.py           # S5: prequential drift + PSI + retraining
│   ├── stats.py           # McNemar's / Wilcoxon significance tests
│   ├── plots.py           # headless matplotlib figure helpers
│   ├── paths.py           # isolated output paths per dataset
│   ├── report.py          # self-contained HTML report assembly
│   ├── pdf_report.py      # publication-quality PDF report generation
│   └── configs/           # packaged configuration templates
├── tests/                 # automated unit & smoke test suite
│   ├── conftest.py        # synthetic dataset fixtures
│   ├── test_config.py     # configuration validation tests
│   ├── test_data.py       # task detection & temporal split tests
│   └── test_pipeline.py   # end-to-end pipeline execution test
├── .github/
│   └── workflows/
│       ├── ci.yml         # GitHub Actions linting & matrix testing
│       └── publish.yml    # PyPI automated release workflow
├── run.py                 # backwards-compatible entry point
├── make_index.py          # backwards-compatible results indexer
└── results/               # generated benchmark reports, models, and figures
```

---

## Engineering notes

A few decisions that keep the numbers honest and the code robust:

- **One evaluation path.** Every model - baseline, booster, stacked, or tuned - is scored through the same `fit_evaluate` in `src/modeling.py`, so leaderboard rows are always comparable.
- **Leakage safety in three places.** The preprocessor is cloned and re-fit inside each pipeline (never fit on the whole dataset); the temporal test set is untouched during tuning and the regularization path (which uses an internal split of the training data only); and the adaptive drift model refits its preprocessor on each window.
- **Cross-validation correctness.** Stacking uses a `StratifiedKFold`/`KFold` partition (scikit-learn's `cross_val_predict` requires one), while Grid/Random tuning can safely use `TimeSeriesSplit` (they use `cross_validate`). This subtlety is handled for you.
- **Single-threaded base learners under parallel search.** Base learners are forced to one thread while the search parallelizes, avoiding cores² oversubscription.
- **Never crashes on a missing optional.** SHAP, CatBoost, Optuna and `river` are all lazily imported; a missing one skips its step with a recorded note.

---

## Documentation

| Document | What's in it |
|---|---|
| **[DATASETS.md](DATASETS.md)** | How to run on other datasets, where to get each CSV, and the insight each run should show. |
| **[PROJECT_PLAN.md](PROJECT_PLAN.md)** | Full design rationale, per-stage technique choices, and the mapping to the source papers. |
| **[EVALUATION_PREP.md](EVALUATION_PREP.md)** | An end-to-end walkthrough of the whole system, plain-English first then technical in depth. |

---

## Literature backing

The design synthesizes five peer-reviewed papers (all from Q1 journals) into one pipeline; each stage traces back to one of them:

| Paper | Journal | Informs |
|---|---|---|
| RegStack | *Expert Systems with Applications* (2025) | Regularized baselines + stacking (S1, S2) |
| Bayesian-Optimized XGBoost | *Expert Systems with Applications* (2026) | The tuning + SHAP backbone (S2–S4) |
| SHAP Error-Compensation | *IEEE Transactions on Artificial Intelligence* (2025) | Reliability-aware explainability (S4) |
| Concept Drift with Continuous Kernel Learning | *Information Sciences* (2024) | Drift monitoring; motivates the ELEC2 dataset (S5) |
| Spectral-Clustering Parameter Tuning | *IEEE Access* (2025) | The "parameters matter as much as algorithms" theme (lit-review) |

See `PROJECT_PLAN.md` for the full stage-by-stage mapping and DOIs.

---

## Roadmap & limitations

- **Compute-first honesty.** The point of the HPO stage is to *measure* the cost/benefit of tuning, not to guarantee tuning always wins - on easy datasets a regularized baseline can match tuned boosters, and the report says so.
- **Monocular drift signals.** PSI and prequential error are strong, cheap drift indicators but do not attribute *why* a distribution moved; deeper causal drift analysis is out of scope.
- **Single-machine scope.** The pipeline targets datasets that fit comfortably in memory on one machine (tens of thousands to low millions of rows); it is not a distributed training system.
- **Possible next steps.** A small pytest suite + CI badge, a CLI flag to select a subset of stages, and optional export of the report to PDF.

---

## License

Released under the **MIT License** - see [LICENSE](LICENSE).

> This began as an Advanced Machine Learning mini-project and is maintained as a reusable tool. If you use it, a link back is appreciated.
