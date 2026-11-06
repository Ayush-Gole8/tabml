# Changelog

All notable changes to TabML-Guard are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-06

### Added
- Standardized package architecture with `pyproject.toml` and `hatchling` build system.
- Unified command line interface (`tabml-guard run / init / index / version`).
- High-level Python programmatic API: `TabMLGuard` class with `.fit()`, `.save_report()`, `.get_best_model()`, and `.predict()`.
- Leakage-safe in-memory `DataBundle` processing (`prepare_bundle`) supporting direct Pandas DataFrames.
- PEP 561 typing marker (`tabml_guard/py.typed`).
- Integrated default and example configuration templates in package resources (`tabml_guard/configs/`).
- Full 6-stage AutoML & MLOps evaluation engine:
  - S0: Data ingestion, profiling, auto-task resolution, temporal splitting.
  - S1: Regularized linear baselines (L1/L2/ElasticNet) and regularization path diagnostics.
  - S2: Boosting ensembles (XGBoost, LightGBM, CatBoost, RandomForest) and regularized Stacking.
  - S3: Cost-accounted hyperparameter optimization comparison (Default vs. Grid vs. Random vs. Bayesian Optuna).
  - S4: SHAP explainability and confidence-based human-in-the-loop reliability triage.
  - S5: Walk-forward prequential drift monitoring, Population Stability Index (PSI), and paired statistical tests (McNemar / Wilcoxon).
  - Self-contained, portable HTML and PDF diagnostic reports.

