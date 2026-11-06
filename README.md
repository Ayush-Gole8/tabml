# TabML-Guard

[![PyPI](https://img.shields.io/pypi/v/tabml-guard?color=blue)](https://pypi.org/project/tabml-guard/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://pypi.org/project/tabml-guard/)
[![License](https://img.shields.io/badge/License-MIT-0f766e)](https://github.com/Ayush-Gole8/tabml/blob/main/LICENSE)
[![CI](https://github.com/Ayush-Gole8/tabml/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayush-Gole8/tabml/actions)

**TabML-Guard** is an explainable, drift-aware AutoML and diagnostic framework designed specifically for time-ordered tabular data.

It automates the full supervised machine learning workflow—from leakage-safe preprocessing and baseline modeling to hyperparameter optimization, SHAP explainability, walk-forward drift simulation, and standalone diagnostic report generation.

---

## Key Features

- **Leakage-Safe Temporal Validation**: Preprocessors are fit strictly on training splits and re-fit within every fold. Time order is preserved to prevent look-ahead bias.
- **Model Diversity**: Trains regularized linear baselines (L1, L2, ElasticNet), gradient boosted trees (XGBoost, LightGBM, CatBoost), Random Forests, and regularized stacking meta-models.
- **Cost-Accounted HPO**: Evaluates Grid, Random, and Bayesian (Optuna TPE) optimization against default configurations under strictly matched evaluation budgets.
- **Explainability & Triage**: Computes global and local SHAP feature attributions paired with uncertainty-calibrated confidence guardrails to route high-risk predictions for human review.
- **Prequential Drift Diagnostics**: Simulates real-world streaming on held-out temporal chunks, comparing frozen vs. adaptive retraining alongside Population Stability Index (PSI) tracking and statistical significance tests (McNemar / Wilcoxon).
- **Zero-Dependency Portable Reporting**: Persists the production pipeline (`best_model.pkl`) and compiles standalone, portable HTML and PDF diagnostic audit reports.

---

## Installation

TabML-Guard requires **Python 3.10 to 3.12**.

```bash
pip install tabml-guard
```

To install with all optional accelerators (such as River for streaming drift detection):

```bash
pip install "tabml-guard[all]"
```

---

## Quickstart

### Python API

```python
import pandas as pd
from tabml_guard import TabMLGuard

# Load your tabular data
df = pd.read_csv("data.csv")

# Initialize and run the diagnostic pipeline
guard = TabMLGuard(
    target="target_column",
    time_col="timestamp",    # Set to None for non-temporal datasets
    task="auto",             # Auto-detects classification vs. regression
    output_dir="results",
)
results = guard.fit(df)

# Export standalone diagnostic audit reports
guard.save_report("report.html", fmt="html")
guard.save_report("report.pdf", fmt="pdf")

# Retrieve the fitted scikit-learn pipeline for inference
model = guard.get_best_model()
predictions = guard.predict(df.drop(columns=["target_column", "timestamp"]))
```

### Command-Line Interface (CLI)

Run TabML-Guard directly on any CSV file:

```bash
tabml-guard run --data data.csv --target target_column --time timestamp
```

Or run using a declarative YAML configuration:

```bash
tabml-guard run --config config.yaml
```

Initialize a starter configuration template in your working directory:

```bash
tabml-guard init
```

---

## Pipeline Stages

TabML-Guard executes a structured 6-stage diagnostic workflow:

| Stage | Name | Description |
|---|---|---|
| **Stage 0** | **Ingestion & Temporal Split** | Ingests data, detects task type, builds leakage-safe transformers, and splits data chronologically. |
| **Stage 1** | **Linear Baselines** | Fits regularized linear models (L1, L2, ElasticNet) and traces regularization paths. |
| **Stage 2** | **Ensembles & Stacking** | Trains boosting models (XGBoost, LightGBM, CatBoost), Random Forest, and a stacking meta-learner. |
| **Stage 3** | **Cost-Accounted HPO** | Compares Default, Grid, Random, and Bayesian search at matched evaluation budgets. |
| **Stage 4** | **Explainability & Triage** | Generates SHAP attributions and calculates confidence-calibrated human review triage curves. |
| **Stage 5** | **Concept Drift & Testing** | Evaluates walk-forward streams, tracks feature PSI over time, and runs statistical significance tests. |

---

## Important Links

- **Source Code**: [github.com/Ayush-Gole8/tabml](https://github.com/Ayush-Gole8/tabml)
- **Issue Tracker**: [github.com/Ayush-Gole8/tabml/issues](https://github.com/Ayush-Gole8/tabml/issues)
- **PyPI Releases**: [pypi.org/project/tabml-guard](https://pypi.org/project/tabml-guard/)

---

## License

TabML-Guard is open-source software released under the [MIT License](https://github.com/Ayush-Gole8/tabml/blob/main/LICENSE).
