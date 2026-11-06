# TabML-Guard

[![PyPI](https://img.shields.io/pypi/v/tabml-guard?color=blue)](https://pypi.org/project/tabml-guard/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://pypi.org/project/tabml-guard/)
[![License](https://img.shields.io/badge/License-MIT-0f766e)](https://github.com/Ayush-Gole8/tabml/blob/main/LICENSE)
[![CI](https://github.com/Ayush-Gole8/tabml/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayush-Gole8/tabml/actions)

**An explainable, drift-aware AutoML pipeline and diagnostic framework for time-ordered tabular data.**

TabML-Guard provides an end-to-end supervised learning and auditing workflow for tabular datasets. Given a tabular dataset (local CSV, OpenML benchmark, or in-memory pandas DataFrame), a target variable, and an optional timestamp, TabML-Guard automates:

- **Task Resolution & Temporal Splitting**: Automatically infers classification vs. regression and enforces strict chronological train/test splits without look-ahead leakage.
- **Regularized Baselines & Path Tracing**: Trains L1, L2, and ElasticNet linear models to establish an honest baseline and traces regularization dynamics.
- **Ensemble Learning & Stacking**: Trains gradient boosted decision trees (XGBoost, LightGBM, CatBoost), Random Forests, and a regularized stacking meta-model.
- **Cost-Accounted Hyperparameter Tuning**: Evaluates Grid, Random, and Bayesian (Optuna TPE) optimization against default configurations under strictly matched evaluation budgets.
- **Explainability & Reliability Guardrails**: Computes global and local SHAP feature attributions and evaluates confidence-calibrated triage to catch failure cases before deployment.
- **Temporal Drift & Prequential Validation**: Simulates real-time walk-forward streaming on held-out test chunks, evaluates frozen vs. sliding-window retraining, tracks Population Stability Index (PSI) covariate shift, and runs formal hypothesis tests (McNemar / Wilcoxon).
- **Self-Contained Diagnostics**: Persists the production pipeline (`best_model.pkl`) and exports portable, standalone HTML and publication-ready PDF audit reports.

TabML-Guard is completely **data-agnostic**: all schema definitions, feature types, target variables, and operational thresholds are specified declaratively via YAML or programmatically via Python.

![TabML-Guard Diagnostic Outputs](https://raw.githubusercontent.com/Ayush-Gole8/tabml/main/assets/sample_outputs.png)

---

## Table of Contents

- [Core Capabilities](#core-capabilities)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Command-Line Interface (CLI)](#1-command-line-interface-cli)
  - [Python Programmatic API](#2-python-programmatic-api)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration Reference](#configuration-reference)
- [Drift & Temporal Evaluation](#drift--temporal-evaluation)
- [Output Artifacts](#output-artifacts)
- [Extensibility](#extensibility)
- [Methodology & References](#methodology--references)
- [License](#license)

---

## Core Capabilities

- **Strict Leakage Prevention**: Preprocessors are fit strictly on training splits and re-fit within every cross-validation fold. Temporal order is preserved to eliminate optimistic data leakage.
- **Compute-Aware Benchmarking**: Quantifies performance gains against wall-clock compute and evaluation trials, revealing whether expensive search procedures justify their cost.
- **Human-in-the-Loop Triage**: Quantifies prediction uncertainty and models selective classification/regression, computing the error-reduction dividend of routing low-confidence cases to manual review.
- **Adaptive Stream Simulation**: Simulates production deployment across sequential time windows, answering whether continuous retraining is statistically necessary.
- **Resilient Fallbacks**: Integrates optional accelerators (CatBoost, Optuna, SHAP, River) via lazy imports with automated fallbacks to keep pipelines operational across diverse environments.

---

## Installation

TabML-Guard supports **Python 3.10–3.12**.

### From PyPI (Standard)
```bash
pip install tabml-guard
```

### With All Optional Accelerators
```bash
pip install "tabml-guard[all]"
```

### From Source (Development)
```bash
git clone https://github.com/Ayush-Gole8/tabml.git
cd tabml
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Command-Line Interface (CLI)

Run TabML-Guard directly on any tabular CSV:
```bash
tabml-guard run --data sales.csv --target revenue --time timestamp
```

Run using a declarative YAML configuration file:
```bash
tabml-guard run --config config.yaml
```

Initialize a template configuration file in the current directory:
```bash
tabml-guard init
```

Index all prior benchmark runs into a consolidated dashboard:
```bash
tabml-guard index --results-dir results
```

---

### 2. Python Programmatic API

TabML-Guard integrates seamlessly into Python workflows and Jupyter notebooks:

```python
import pandas as pd
from tabml_guard import TabMLGuard

# Load tabular data
df = pd.read_csv("customer_data.csv")

# Initialize the automated pipeline
guard = TabMLGuard(
    target="churn",
    time_col="signup_date",  # Set to None for standard non-temporal validation
    task="auto",             # Auto-detects classification vs. regression
    output_dir="audit_results",
)

# Run end-to-end execution
results = guard.fit(df)

# Export standalone diagnostic reports
guard.save_report("churn_audit.html", fmt="html")
guard.save_report("churn_audit.pdf", fmt="pdf")

# Retrieve production scikit-learn pipeline for inference
model = guard.get_best_model()
predictions = guard.predict(df.drop(columns=["churn", "signup_date"]))
```

---

## Pipeline Architecture

TabML-Guard executes a coordinated 6-stage diagnostic flow, passing validated intermediate artifacts through memory:

```text
[ Raw Tabular Data: CSV / OpenML / DataFrame ]
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 0 · Ingestion, Profiling & Temporal Splitting                    │
│ • Detects task type (Binary, Multiclass, Regression) from target       │
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

### Stage Summary

| Stage | Operation | Objective |
|---|---|---|
| **S0 · Ingestion & Split** | Ingests data, profiles column distributions, infers task type, builds preprocessor, and chronologically splits. | Prevents future look-ahead leakage and establishes baseline integrity. |
| **S1 · Baselines** | Fits regularized linear models (L1, L2, ElasticNet) and traces regularization paths. | Establishes an interpretable, parsimonious performance benchmark. |
| **S2 · Ensembles** | Trains XGBoost, LightGBM, CatBoost, Random Forest, and a cross-validated stacking meta-learner. | Discovers non-linear feature interactions and high-capacity ensemble combinations. |
| **S3 · Cost-Accounted HPO** | Compares Default, Grid, Random, and Bayesian search at equal evaluation budgets. | Quantifies marginal performance gain relative to compute duration. |
| **S4 · Explainability & Triage** | Generates SHAP global and local attributions and models selective prediction confidence. | Provides feature transparency and catches errors via human-in-the-loop triage. |
| **S5 · Drift & Significance** | Replays held-out test chunks sequentially, evaluates frozen vs. adaptive retraining, and tracks PSI. | Diagnoses model degradation under temporal covariate shift. |

---

## Configuration Reference

Pipelines can be configured declaratively via YAML:

```yaml
dataset:
  source: csv                     # "csv" or "openml"
  path: "data/my_dataset.csv"     # Path to CSV
  target: "target_col"            # Target variable column
  time_col: "timestamp"           # Chronological ordering column (or null)
  drop_cols: ["id", "leakage"]    # Columns to exclude

task: auto                        # "auto", "classification", or "regression"

split:
  train_frac: 0.6                 # Fraction allocated to initial training
  test_frac: 0.4                  # Fraction allocated to held-out test stream

models:
  boosters:
    - xgboost
    - lightgbm
    - catboost
    - random_forest
  stacking: true                  # Enable regularized meta-learner stacking

tuning:
  budget_trials: 40               # Budget for HPO search strategies
  cv_splits: 3                    # Cross-validation folds
  scoring: null                   # Metric (null auto-assigns ROC-AUC or neg-RMSE)

drift:
  n_chunks: 10                    # Number of sequential test chunks
  retrain: sliding                # "none", "periodic", or "sliding"

report:
  title: "TabML-Guard Audit Report"
  fbeta: 2.0                      # Beta parameter for F-beta score

output_dir: "results"             # Output directory for models and reports
random_state: 42                  # Deterministic random seed
```

---

## Drift & Temporal Evaluation

Standard cross-validation shuffles data across time, generating overly optimistic performance estimates on real-world time-series tabular data. TabML-Guard enforces strict prequential validation:

```text
[ Initial Training Window: 60% ] ──> [ Held-Out Sequential Stream: 40% ]
                                                  │
                                                  ├──> Slice 1 (Test ──> Retrain)
                                                  ├──> Slice 2 (Test ──> Retrain)
                                                  └──> Slice N (Test ──> Retrain)
```

1. **Frozen Baseline**: A model trained strictly on historical data and evaluated forward without adaptation.
2. **Adaptive Retraining**: Evaluates predictions on the incoming chunk *before* incorporating observations into the updated training window.
3. **Population Stability Index (PSI)**: Monitors continuous covariate shift per feature over time:
   $$\text{PSI} = \sum \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$
4. **Hypothesis Testing**: Runs paired McNemar's tests (for classification) or Wilcoxon signed-rank tests (for regression) to determine if performance differences between models are statistically significant ($p < 0.05$).

![Prequential Drift Performance](https://raw.githubusercontent.com/Ayush-Gole8/tabml/main/assets/drift_performance.png)

---

## Output Artifacts

Execution writes structured artifacts into the designated `output_dir`:

```
results/
├── report_<dataset>.html    # Self-contained, portable HTML diagnostic report
├── report_<dataset>.pdf     # Publication-ready vector PDF report
├── best_model.pkl           # Trained scikit-learn pipeline (ready for production inference)
├── tables/
│   ├── profile.json         # Dataset schema, missingness, and cardinality profile
│   └── results.json         # Structured JSON metrics and model parameters across all stages
└── figures/
    ├── shap_importance.png  # SHAP global feature importances
    ├── reliability.png      # Error capture vs. coverage triage curve
    ├── drift_performance.png# Frozen vs. adaptive streaming accuracy
    └── drift_psi.png        # Feature-level PSI trajectory
```

Reports embed all figures and styling inline, requiring no external dependencies, web servers, or active internet connection to view.

---

## Extensibility

TabML-Guard is designed modularly to support custom extensions:

- **Custom Estimators**: Register new scikit-learn compatible classifiers or regressors in `tabml_guard/boosting.py`.
- **Custom Search Spaces**: Define parameter distributions for novel models in `tabml_guard/tuning.py`.
- **Custom Metrics**: Register task-specific scoring functions in `tabml_guard/metrics.py`.
- **Drift Detectors**: Add alternative streaming change detectors in `tabml_guard/drift.py`.

---

## Methodology & References

The diagnostic design of TabML-Guard incorporates methodology from the following literature:

1. **RegStack Architecture & Regularized Baselines**  
   *Expert Systems with Applications* (2025). Formulates the benchmark framework contrasting regularized linear paths (L1/L2/ElasticNet) against multi-layer stacking meta-models.
2. **Cost-Accounted Bayesian Optimization & Tree Boosting**  
   *Expert Systems with Applications* (2026). Informs the compute-bounded comparison of Tree-structured Parzen Estimators (TPE) against random search regimes.
3. **Reliability-Aware Model Explainability**  
   *IEEE Transactions on Artificial Intelligence* (2025). Motivates coupling SHAP feature attributions with uncertainty-calibrated triage guardrails.
4. **Prequential Concept Drift & Stream Diagnostics**  
   *Information Sciences* (2024). Establishes walk-forward evaluation protocols and covariate shift monitoring under temporal drift.

---

## License

TabML-Guard is open-source software licensed under the **[MIT License](https://github.com/Ayush-Gole8/tabml/blob/main/LICENSE)**.
