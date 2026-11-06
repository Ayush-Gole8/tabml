"""Shared model training & evaluation core used by every stage.

Keeping *one* fit/evaluate path here (instead of re-implementing it in baselines,
boosting, tuning, ...) guarantees every row on the leaderboard is produced the
same way and is therefore directly comparable. It also centralizes the tricky
bits — leakage-safe preprocessing, probability extraction for AUC, and
nan-robust "which model won?" logic — so each stage stays small.

Design contract with the rest of the system:
  * Preprocessing lives INSIDE each model's Pipeline and is cloned per fit, so
    imputation/scaling statistics are learned from training folds only.
  * A ``ModelResult`` carries the fitted estimator plus arrays (for downstream
    stages) AND a JSON-safe ``summary()`` (for the report). Fitted models travel
    between stages in an in-memory ``ctx`` dict; only ``summary()`` output is
    ever serialized.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from .config import Config
from .data import DataBundle
from .metrics import compute_metrics, higher_is_better, primary_metric_name


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    """Everything produced by training one model.

    ``estimator``/``y_pred``/``y_proba`` are in-memory artifacts for later stages
    (SHAP, drift, McNemar's). ``summary()`` is the JSON-safe view for the report.
    """

    name: str
    estimator: Any                      # fitted sklearn Pipeline (pre + model)
    metrics: dict
    fit_seconds: float
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)   # JSON-safe diagnostics only

    def summary(self) -> dict:
        row = {
            "name": self.name,
            "metrics": self.metrics,
            "fit_seconds": round(float(self.fit_seconds), 3),
        }
        if self.extra:
            row["extra"] = self.extra
        return row


# --------------------------------------------------------------------------- #
# Pipeline assembly & prediction helpers
# --------------------------------------------------------------------------- #
def make_pipeline(preprocessor, estimator) -> Pipeline:
    """Wrap an estimator behind a *cloned* copy of the shared preprocessor.

    Cloning matters: the ``bundle.preprocessor`` is an unfitted template shared by
    every model. Without cloning, models would share (and refit) one transformer
    object, which is both a correctness hazard and a subtle source of leakage.
    """
    return Pipeline([("pre", clone(preprocessor)), ("model", estimator)])


def feature_names_out(fitted_pipeline) -> list[str]:
    """Readable feature names after preprocessing (best effort; [] if unavailable)."""
    try:
        pre = fitted_pipeline.named_steps["pre"]
        return [str(n) for n in pre.get_feature_names_out()]
    except Exception:
        return []


def get_proba(fitted_pipeline, X, task: str) -> Optional[np.ndarray]:
    """Class probabilities (or decision scores) for AUC/PR-AUC.

    Returns None for regression or when the model exposes neither
    ``predict_proba`` nor ``decision_function``. ``metrics`` tolerates both a
    1-D positive-class score and a full (n, n_classes) matrix.
    """
    if task != "classification":
        return None
    if hasattr(fitted_pipeline, "predict_proba"):
        try:
            return np.asarray(fitted_pipeline.predict_proba(X))
        except Exception:
            pass
    if hasattr(fitted_pipeline, "decision_function"):
        try:
            return np.asarray(fitted_pipeline.decision_function(X))
        except Exception:
            pass
    return None


def fit_evaluate(name: str, estimator, bundle: DataBundle, cfg: Config) -> ModelResult:
    """Fit ``estimator`` (behind the preprocessor) on train and score on test.

    THE single place a model is trained and scored. Every stage calls this so the
    leaderboard is apples-to-apples.
    """
    pipe = make_pipeline(bundle.preprocessor, estimator)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # Regularized linear models on many rows can hit the iteration cap and
        # emit a ConvergenceWarning; we raise max_iter generously elsewhere, so
        # silence just this class to keep the console demo-readable.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        pipe.fit(bundle.X_train, bundle.y_train)
    fit_seconds = time.perf_counter() - t0

    y_pred = np.asarray(pipe.predict(bundle.X_test)).ravel()
    if bundle.task == "classification":
        # Targets are LabelEncoded 0..K-1; normalize prediction dtype so metrics
        # never trip over float labels (e.g. CatBoost returns float class ids).
        y_pred = y_pred.astype(int)

    n_classes = len(bundle.target_classes) if bundle.target_classes else 2
    y_proba = get_proba(pipe, bundle.X_test, bundle.task)
    metrics = compute_metrics(
        bundle.task,
        bundle.y_test,
        y_pred,
        y_proba,
        beta=cfg.report.fbeta,
        n_classes=n_classes,
    )
    return ModelResult(
        name=name,
        estimator=pipe,
        metrics=metrics,
        fit_seconds=fit_seconds,
        y_pred=y_pred,
        y_proba=None if y_proba is None else np.asarray(y_proba),
    )


# --------------------------------------------------------------------------- #
# Ranking & logging
# --------------------------------------------------------------------------- #
def pick_best(results: list[ModelResult], task: str) -> Optional[ModelResult]:
    """Return the result with the best primary metric (nan-safe).

    Uses ``primary_metric_name`` (roc_auc for classification, rmse for
    regression) and ``higher_is_better`` so ranking is correct in both
    directions. Models whose primary metric is nan sort to the bottom.
    """
    if not results:
        return None
    metric = primary_metric_name(task)
    hib = higher_is_better(metric)
    worst = float("-inf")

    def score(r: ModelResult) -> float:
        v = r.metrics.get(metric, float("nan"))
        if v != v:  # nan
            return worst
        return v if hib else -v

    return max(results, key=score)


def log_result(stage: str, res: ModelResult, task: str, fbeta: float) -> None:
    """One tidy console line per model — the same layout for every stage."""
    m = res.metrics
    if task == "classification":
        fb = f"f{fbeta:g}"
        print(
            f"[{stage}] {res.name:<20} "
            f"acc={m.get('accuracy', float('nan')):.4f}  "
            f"f1={m.get('f1', float('nan')):.4f}  "
            f"{fb}={m.get(fb, float('nan')):.4f}  "
            f"roc_auc={m.get('roc_auc', float('nan')):.4f}  "
            f"pr_auc={m.get('pr_auc', float('nan')):.4f}  "
            f"({res.fit_seconds:.1f}s)"
        )
    else:
        print(
            f"[{stage}] {res.name:<20} "
            f"rmse={m.get('rmse', float('nan')):.4f}  "
            f"mae={m.get('mae', float('nan')):.4f}  "
            f"r2={m.get('r2', float('nan')):.4f}  "
            f"({res.fit_seconds:.1f}s)"
        )


def leaderboard(results: list[ModelResult], task: str) -> list[dict]:
    """JSON-safe, best-first ranking across any set of results (for the report)."""
    metric = primary_metric_name(task)
    hib = higher_is_better(metric)

    def key(d: dict) -> float:
        v = d["primary"]
        if v != v:  # nan
            return float("-inf")
        return v if hib else -v

    rows = [
        {
            "name": r.name,
            "primary": r.metrics.get(metric, float("nan")),
            "metrics": r.metrics,
            "fit_seconds": round(float(r.fit_seconds), 3),
        }
        for r in results
    ]
    rows.sort(key=key, reverse=True)
    return rows
