"""High-level Python API for TabML-Guard.

Provides the ``TabMLGuard`` class for programmatic pipeline execution on
Pandas DataFrames, CSVs, or OpenML datasets without requiring CLI usage.

Example
-------
>>> from tabml_guard import TabMLGuard
>>> import pandas as pd
>>> df = pd.read_csv("my_data.csv")
>>> guard = TabMLGuard(target="churn", time_col="timestamp")
>>> results = guard.fit(df)
>>> html_path = guard.save_report(fmt="html")
>>> model = guard.get_best_model()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from . import baselines, boosting, drift, explain, paths, pdf_report, report, tuning
from .config import (
    Config,
    DatasetConfig,
    DriftConfig,
    ModelsConfig,
    ReportConfig,
    SplitConfig,
    TuningConfig,
    load_config,
)
from .data import DataBundle, prepare, prepare_bundle


class TabMLGuard:
    """An explainable, drift-aware AutoML pipeline for time-ordered tabular data.

    Parameters
    ----------
    target : str, optional
        Target column name to predict.
    time_col : str, optional
        Column giving temporal order. If None, performs random split and skips drift.
    task : str, default="auto"
        Task type: ``'auto'``, ``'classification'``, or ``'regression'``.
    output_dir : str, default="results"
        Directory where reports, figures, and models are written.
    drop_cols : list of str, optional
        Columns to exclude (IDs, leakage, free text).
    budget_trials : int, default=40
        Hyperparameter optimization budget (number of Optuna/Random trials).
    cv_splits : int, default=3
        Cross-validation splits for tuning and evaluation.
    config : Config or str or Path, optional
        Pre-built Config object or path to YAML config file.
    """

    def __init__(
        self,
        target: str = "",
        time_col: Optional[str] = None,
        task: str = "auto",
        output_dir: str = "results",
        drop_cols: Optional[list[str]] = None,
        budget_trials: int = 40,
        cv_splits: int = 3,
        config: Optional[Union[Config, str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            if isinstance(config, (str, Path)):
                self.config = load_config(config)
            elif isinstance(config, Config):
                self.config = config
            else:
                raise TypeError(f"config must be Config, str, or Path, got {type(config).__name__}")
        else:
            self.config = Config(
                dataset=DatasetConfig(
                    source="csv",
                    target=target,
                    time_col=time_col,
                    drop_cols=drop_cols or [],
                ),
                task=task,
                output_dir=output_dir,
                tuning=TuningConfig(budget_trials=budget_trials, cv_splits=cv_splits),
                report=ReportConfig(title=kwargs.get("title", f"TabML-Guard — {target or 'Report'}")),
            )

        self.bundle_: Optional[DataBundle] = None
        self.context_: dict[str, Any] = {}
        self.report_context_: dict[str, Any] = {}
        self.html_report_path_: Optional[str] = None
        self.pdf_report_path_: Optional[str] = None

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> TabMLGuard:
        """Instantiate TabMLGuard from a YAML configuration file."""
        return cls(config=load_config(path))

    def fit(self, df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
        """Execute the full S0→S5 diagnostic and training pipeline.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Input DataFrame. If None, loads from the configured dataset source (CSV/OpenML).

        Returns
        -------
        dict
            JSON-safe summary results of all pipeline stages.
        """
        cfg = self.config
        run_out = paths.run_dir(cfg)

        # Stage 0: Data Preparation
        if df is not None:
            self.bundle_ = prepare_bundle(df, cfg)
        else:
            self.bundle_ = prepare(cfg)

        bundle = self.bundle_
        (paths.tables_dir(cfg) / "profile.json").write_text(
            json.dumps(bundle.profile, indent=2), encoding="utf-8"
        )

        ctx: dict[str, Any] = {}
        report_ctx: dict[str, Any] = {"profile": bundle.profile, "task": bundle.task}

        # Stage 1: Regularized Baselines
        report_ctx["baselines"] = baselines.run(bundle, cfg, ctx)

        # Stage 2: Ensembles & Stacking
        report_ctx["boosting"] = boosting.run(bundle, cfg, ctx)

        # Stage 3: HPO Comparison
        report_ctx["tuning"] = tuning.run(bundle, cfg, ctx)

        # Stage 4: Explainability & Triage
        report_ctx["explain"] = explain.run(bundle, cfg, ctx)

        # Stage 5: Drift & Statistical Significance
        report_ctx["drift"] = drift.run(bundle, cfg, ctx)

        # Persist champion model & JSON summary
        best = ctx.get("best_overall")
        if best is not None:
            try:
                import joblib

                joblib.dump(best.estimator, paths.model_path(cfg))
            except Exception as exc:
                print(f"[pipeline] could not persist best model: {exc}")

        (paths.tables_dir(cfg) / "results.json").write_text(
            json.dumps(report_ctx, indent=2), encoding="utf-8"
        )

        # Stage 6: Build Reports (Both PDF and HTML)
        self.pdf_report_path_ = pdf_report.build(report_ctx, str(run_out), cfg.report.title, cfg=cfg)
        self.html_report_path_ = report.build(report_ctx, str(run_out), cfg.report.title, cfg=cfg)

        self.context_ = ctx
        self.report_context_ = report_ctx
        return report_ctx

    def save_report(self, path: Optional[Union[str, Path]] = None, fmt: str = "html") -> str:
        """Return or export the generated audit report.

        Parameters
        ----------
        path : str or Path, optional
            Destination path. If None, returns the generated report path in output_dir.
        fmt : {'html', 'pdf'}, default='html'
            Format of the report to save.
        """
        if not self.report_context_:
            raise RuntimeError("Pipeline must be fitted before saving reports. Call .fit() first.")

        src = self.html_report_path_ if fmt.lower() == "html" else self.pdf_report_path_
        if src is None:
            raise FileNotFoundError(f"Report for format '{fmt}' was not generated.")

        if path is not None:
            import shutil

            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            return str(dest)
        return src

    def get_best_model(self) -> Any:
        """Return the champion fitted scikit-learn Pipeline (preprocessor + model)."""
        best = self.context_.get("best_overall")
        if best is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call .fit() first.")
        return best.estimator

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Return the ranked model leaderboard across all stages."""
        for stage in ("tuning", "boosting", "baselines"):
            lb = (self.report_context_.get(stage) or {}).get("leaderboard")
            if lb:
                return lb
        return []

    def get_profile(self) -> dict[str, Any]:
        """Return dataset profiling diagnostics."""
        return self.report_context_.get("profile", {})

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict target using the fitted champion pipeline."""
        model = self.get_best_model()
        return model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Predict class probabilities using the fitted champion pipeline."""
        model = self.get_best_model()
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        raise AttributeError(f"Champion model '{type(model).__name__}' does not support predict_proba.")

