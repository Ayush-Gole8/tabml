"""Output-path resolution for TabML-Guard.

Every run writes into its own per-dataset directory::

    <output_dir>/<slug>/
        report_<slug>.pdf
        best_model.pkl
        figures/*.png
        tables/{profile,results}.json

so running several configs leaves one self-contained folder per dataset instead
of overwriting a single shared ``figures/`` directory.

The slug is derived from the config *before any stage runs*, because Stage 4 and
Stage 5 write figures long before the report is assembled. ``run_dir(cfg)`` is
therefore the single source of truth that every writer consults.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


def dataset_slug(cfg: Optional[Any] = None, context: Optional[dict] = None,
                 title: str = "") -> str:
    """A filesystem-safe dataset identifier.

    Prefers the configured OpenML name, else the CSV file stem, else the OpenML
    data_id, else the target column, else the title. Guarantees a non-empty slug.

    The precedence is deliberately unchanged from the original implementation in
    ``report.py`` so existing report filenames stay stable.
    """
    raw: Optional[str] = None
    if cfg is not None:
        try:
            ds = cfg.dataset
            if getattr(ds, "name", None):
                raw = str(ds.name)
            elif getattr(ds, "path", None):
                raw = Path(str(ds.path)).stem
            elif getattr(ds, "data_id", None):
                raw = f"openml{ds.data_id}"
        except Exception:
            raw = None
    if not raw:
        prof = (context or {}).get("profile", {}) or {}
        raw = prof.get("target") or title or "report"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(raw)).strip("-").lower()
    return slug or "dataset"


def run_dir(cfg: Any) -> Path:
    """The per-dataset output directory for this run (created on demand)."""
    d = Path(cfg.output_dir) / dataset_slug(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(cfg: Any) -> Path:
    d = run_dir(cfg) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir(cfg: Any) -> Path:
    d = run_dir(cfg) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_path(cfg: Any, ext: str = "pdf") -> Path:
    """Where the report for this run belongs, e.g. results/electricity/report_electricity.pdf"""
    slug = dataset_slug(cfg)
    return run_dir(cfg) / f"report_{slug}.{ext}"


def model_path(cfg: Any) -> Path:
    return run_dir(cfg) / "best_model.pkl"
