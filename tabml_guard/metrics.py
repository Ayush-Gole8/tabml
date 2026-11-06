"""Task-aware metric computation shared across all stages.

Keeping metrics in one place means every model in the leaderboard is scored the
same way, and the report can trust the numbers are comparable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    beta: float = 2.0,
    n_classes: int = 2,
) -> dict:
    """Precision/Recall/F1/F-beta + ROC-AUC/PR-AUC when probabilities are available.

    Handles both binary and multiclass. For multiclass, AUC/PR-AUC use macro/OVR
    averaging and probabilities must be a (n_samples, n_classes) array.
    """
    avg = "binary" if n_classes == 2 else "macro"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        f"f{beta:g}": float(fbeta_score(y_true, y_pred, beta=beta, average=avg, zero_division=0)),
    }

    if y_proba is not None:
        try:
            if n_classes == 2:
                # Accept either a 1-D positive-class prob or a 2-column proba matrix.
                pos = y_proba[:, 1] if getattr(y_proba, "ndim", 1) == 2 else y_proba
                out["roc_auc"] = float(roc_auc_score(y_true, pos))
                out["pr_auc"] = float(average_precision_score(y_true, pos))
            else:
                out["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
                out["pr_auc"] = float(
                    average_precision_score(y_true, _one_hot(y_true, n_classes), average="macro")
                )
        except (ValueError, IndexError):
            # e.g. a class missing from a fold; skip AUC rather than crash.
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    beta: float = 2.0,
    n_classes: int = 2,
) -> dict:
    """Dispatch to the right metric set for the task."""
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_proba, beta=beta, n_classes=n_classes)
    return regression_metrics(y_true, y_pred)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def primary_metric_name(task: str) -> str:
    """The single headline metric used to rank the leaderboard."""
    return "roc_auc" if task == "classification" else "rmse"


def higher_is_better(metric: str) -> bool:
    return metric not in {"rmse", "mae"}


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(y), n_classes), dtype=int)
    oh[np.arange(len(y)), y.astype(int)] = 1
    return oh
