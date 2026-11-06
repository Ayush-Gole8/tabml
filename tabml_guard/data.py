"""Data ingestion, profiling, task detection, preprocessing, and temporal split.

This is the foundation every later stage builds on. It is deliberately
dataset-agnostic: nothing here knows about "electricity" specifically — all
dataset details come from the config. That is what makes the tool reusable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .config import Config


# --------------------------------------------------------------------------- #
# The object passed to every downstream stage
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    task: str                                   # 'classification' | 'regression'
    feature_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    preprocessor: ColumnTransformer             # UNFITTED template; clone before fitting
    target_classes: Optional[list] = None       # class labels (classification only)
    time_train: Optional[pd.Series] = None
    time_test: Optional[pd.Series] = None
    has_time: bool = False
    profile: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(cfg: Config) -> pd.DataFrame:
    """Load the raw dataset into a single DataFrame (features + target together)."""
    ds = cfg.dataset
    if ds.source == "openml":
        from sklearn.datasets import fetch_openml

        if ds.data_id:
            print(f"[data] Fetching OpenML dataset by id {ds.data_id}...")
            bunch = fetch_openml(data_id=int(ds.data_id), as_frame=True, parser="auto")
        else:
            print(f"[data] Fetching OpenML dataset '{ds.name}' (version {ds.version})...")
            bunch = fetch_openml(
                name=str(ds.name), version=ds.version, as_frame=True, parser="auto"
            )
        df = bunch.frame.copy()
    else:  # csv
        path = Path(ds.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        print(f"[data] Reading CSV '{path}'...")
        df = pd.read_csv(path)

    if ds.target not in df.columns:
        raise KeyError(
            f"target column '{ds.target}' not in dataset. Available: {list(df.columns)}"
        )
    if ds.time_col and ds.time_col not in df.columns:
        raise KeyError(
            f"time_col '{ds.time_col}' not in dataset. Available: {list(df.columns)}"
        )
    missing_drop = [c for c in ds.drop_cols if c not in df.columns]
    if missing_drop:
        print(f"[data] NOTE: drop_cols not found (ignored): {missing_drop}. "
              f"Available: {list(df.columns)}")
    print(f"[data] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df


# --------------------------------------------------------------------------- #
# Task detection
# --------------------------------------------------------------------------- #
def resolve_task(y: pd.Series, task_cfg: str) -> str:
    """Decide classification vs regression, honoring an explicit override."""
    if task_cfg in {"classification", "regression"}:
        return task_cfg

    # auto-detect
    if y.dtype.kind in "OSb" or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    n_unique = int(y.nunique(dropna=True))
    if y.dtype.kind in "iuf" and n_unique <= 20 and (n_unique / max(len(y), 1)) < 0.05:
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """A leakage-safe preprocessor: impute+scale numerics, impute+one-hot categoricals.

    Returned UNFITTED so each model pipeline fits it on its own training fold.
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Scalability guard: with sparse_output=False, a very high-cardinality column
    # expands into that many dense columns and can blow up memory. Warn early so
    # the user can group rare levels or drop the column, rather than OOM mid-run.
    for c in categorical_cols:
        n_unique = int(X[c].nunique(dropna=True))
        if n_unique > 50:
            print(
                f"[data] WARNING: categorical '{c}' has {n_unique} unique values; "
                f"dense one-hot will add ~{n_unique} columns. "
                f"Consider grouping rare levels or treating it as an ID to drop."
            )

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_dataframe(
    df: pd.DataFrame, target: str, task: str, time_col: Optional[str]
) -> dict:
    """A compact, JSON-serializable snapshot of the dataset for the report."""
    y = df[target]
    prof: dict = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": target,
        "task": task,
        "n_features": int(df.shape[1] - 1 - (1 if time_col else 0)),
        "missing_total": int(df.isna().sum().sum()),
        "columns_with_missing": {
            col: int(n) for col, n in df.isna().sum().items() if n > 0
        },
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
    }
    if task == "classification":
        counts = y.value_counts(dropna=False)
        prof["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        prof["n_classes"] = int(y.nunique(dropna=True))
        prof["class_balance"] = {
            str(k): round(float(v) / len(y), 4) for k, v in counts.items()
        }
    else:
        prof["target_stats"] = {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "std": float(y.std()),
        }
    if time_col:
        prof["time_col"] = time_col
        prof["time_span"] = [str(df[time_col].min()), str(df[time_col].max())]
    return prof


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _temporal_split_indices(n: int, train_frac: float, test_frac: float):
    """First `train_frac` rows -> train; last `test_frac` rows -> test (by position)."""
    n_train = int(round(n * train_frac))
    n_test = int(round(n * test_frac))
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n - n_test, n)
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def prepare_bundle(df: pd.DataFrame, cfg: Config) -> DataBundle:
    """Detect task -> profile -> split (temporal or random) -> package DataBundle from an in-memory DataFrame."""
    ds = cfg.dataset

    # Order by time if we have a time column, so the split is truly chronological.
    has_time = bool(ds.time_col)
    if has_time:
        df = df.sort_values(by=ds.time_col, kind="mergesort").reset_index(drop=True)

    y_series = df[ds.target]
    task = resolve_task(y_series, cfg.task)
    profile = profile_dataframe(df, ds.target, task, ds.time_col)
    print(f"[data] Task resolved to: {task.upper()}")

    # Build the feature matrix. Drop the target; drop the time column unless
    # explicitly kept; drop any user-listed columns (IDs / leakage / free text).
    to_drop = [ds.target]
    time_all = None
    if has_time:
        time_all = df[ds.time_col].reset_index(drop=True)
        if not ds.use_time_as_feature:
            to_drop.append(ds.time_col)
    to_drop += [c for c in ds.drop_cols if c in df.columns and c not in to_drop]
    X = df.drop(columns=to_drop)

    # Reflect the true feature count (after all drops) in the profile.
    profile["n_features"] = int(X.shape[1])

    # Encode target for classification (0..K-1); keep raw values for regression.
    target_classes = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_series.astype(str))
        target_classes = list(le.classes_)
    else:
        y = y_series.to_numpy(dtype=float)

    # Split.
    if has_time:
        tr, te = _temporal_split_indices(len(X), cfg.split.train_frac, cfg.split.test_frac)
        X_train, X_test = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        y_train, y_test = y[tr], y[te]
        time_train = time_all.iloc[tr].reset_index(drop=True)
        time_test = time_all.iloc[te].reset_index(drop=True)
        print(
            f"[data] Temporal split: train={len(X_train):,} (earliest) | "
            f"test={len(X_test):,} (latest)."
        )
    else:
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.split.test_frac,
            random_state=cfg.random_state,
            stratify=stratify,
            shuffle=True,
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        time_train = time_test = None
        print(
            f"[data] No time column -> random split "
            f"(train={len(X_train):,} | test={len(X_test):,}). Drift analysis will be skipped."
        )

    pre, num_cols, cat_cols = build_preprocessor(X)
    print(
        f"[data] Features: {len(X.columns)} total "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical)."
    )

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task=task,
        feature_names=list(X.columns),
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        preprocessor=pre,
        target_classes=target_classes,
        time_train=time_train,
        time_test=time_test,
        has_time=has_time,
        profile=profile,
    )


def prepare(cfg: Config) -> DataBundle:
    """Load from source -> detect task -> profile -> split -> package."""
    df = load_raw(cfg)
    return prepare_bundle(df, cfg)
