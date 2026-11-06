"""Configuration loading and validation for TabML-Guard.

The whole pipeline is driven by a single YAML file. This module turns that YAML
into a set of typed dataclasses with sensible defaults and validates it, so the
rest of the code can rely on a clean, predictable config object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------- #
# Typed config sections
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    source: str = "openml"                 # 'openml' | 'csv'
    name: Optional[str] = None             # OpenML dataset name/id (as string)
    data_id: Optional[int] = None          # OpenML numeric id (canonical; overrides name)
    path: Optional[str] = None             # local CSV path (when source == 'csv')
    version: int = 1                       # OpenML version (ignored when data_id is set)
    target: str = ""                       # column to predict
    time_col: Optional[str] = None         # column giving temporal order (or None)
    use_time_as_feature: bool = False      # keep time_col as a model feature?
    drop_cols: list = field(              # columns to exclude (IDs, leakage, free text)
        default_factory=list
    )


@dataclass
class SplitConfig:
    train_frac: float = 0.6
    test_frac: float = 0.4


@dataclass
class ModelsConfig:
    boosters: list = field(
        default_factory=lambda: ["xgboost", "lightgbm", "catboost", "random_forest"]
    )
    stacking: bool = True


@dataclass
class TuningConfig:
    budget_trials: int = 40
    cv_splits: int = 3
    scoring: Optional[str] = None          # None -> chosen automatically by task


@dataclass
class DriftConfig:
    n_chunks: int = 10
    retrain: str = "sliding"               # 'none' | 'periodic' | 'sliding'


@dataclass
class ReportConfig:
    title: str = "TabML-Guard Report"
    fbeta: float = 2.0


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    task: str = "auto"                     # 'auto' | 'classification' | 'regression'
    random_state: int = 42
    output_dir: str = "results"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading / validation
# --------------------------------------------------------------------------- #
def _section(raw: dict, key: str) -> dict:
    """Return a sub-dict for a config section, tolerating missing/None."""
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping, got {type(value).__name__}.")
    return value


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top level of the config file must be a mapping.")

    cfg = Config(
        dataset=DatasetConfig(**_section(raw, "dataset")),
        split=SplitConfig(**_section(raw, "split")),
        models=ModelsConfig(**_section(raw, "models")),
        tuning=TuningConfig(**_section(raw, "tuning")),
        drift=DriftConfig(**_section(raw, "drift")),
        report=ReportConfig(**_section(raw, "report")),
        task=raw.get("task", "auto"),
        random_state=int(raw.get("random_state", 42)),
        output_dir=raw.get("output_dir", "results"),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail fast on obviously-wrong configs, with actionable messages."""
    ds = cfg.dataset

    if ds.source not in {"openml", "csv"}:
        raise ValueError(f"dataset.source must be 'openml' or 'csv', got '{ds.source}'.")
    if ds.source == "openml" and not ds.name and not ds.data_id:
        raise ValueError("dataset.name or dataset.data_id is required when dataset.source is 'openml'.")
    if ds.source == "csv" and not ds.path:
        raise ValueError("dataset.path is required when dataset.source is 'csv'.")
    if not ds.target:
        raise ValueError("dataset.target (the column to predict) must be set.")

    if cfg.task not in {"auto", "classification", "regression"}:
        raise ValueError("task must be 'auto', 'classification', or 'regression'.")

    tf, ef = cfg.split.train_frac, cfg.split.test_frac
    if not (0 < tf < 1) or not (0 < ef < 1):
        raise ValueError("split.train_frac and split.test_frac must each be between 0 and 1.")
    if tf + ef > 1.0 + 1e-9:
        raise ValueError(f"split.train_frac + split.test_frac must be <= 1 (got {tf + ef:.3f}).")

    if cfg.drift.retrain not in {"none", "periodic", "sliding"}:
        raise ValueError("drift.retrain must be 'none', 'periodic', or 'sliding'.")
    if cfg.drift.n_chunks < 2:
        raise ValueError("drift.n_chunks must be at least 2.")
    if cfg.tuning.budget_trials < 1:
        raise ValueError("tuning.budget_trials must be >= 1.")
    if cfg.tuning.cv_splits < 2:
        raise ValueError("tuning.cv_splits must be >= 2.")
