"""Plotting helpers. Every figure is saved as a PNG into the run's figures
directory and its path returned, so the report can embed it.

Stages call these with ``output_dir`` being the *run directory* for this dataset
(``results/<slug>/``), so figures from different configs never overwrite one
another. See ``src/paths.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt  # noqa: E402


def figures_dir(output_dir: str) -> Path:
    d = Path(output_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, output_dir: str, name: str) -> str:
    """Save a matplotlib figure and return its path (as a string)."""
    path = figures_dir(output_dir) / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Generic building blocks (each stage composes these)
# --------------------------------------------------------------------------- #
def plot_barh(names, values, output_dir: str, name: str,
              title: str = "", xlabel: str = "") -> str:
    """Horizontal bar chart (feature importances, etc.). Assumes pre-sorted input."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(names))))
    ax.barh(range(len(names)), values, color="#3b6ea5")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    return save_fig(fig, output_dir, name)


def plot_reliability(mean_conf, accuracy, output_dir: str, name: str) -> str:
    """Reliability/calibration curve: model confidence vs. observed accuracy."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfectly calibrated")
    ax.plot(mean_conf, accuracy, "o-", color="#c0504d", label="model")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability curve", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return save_fig(fig, output_dir, name)


def plot_lines(x, series: dict, output_dir: str, name: str,
               title: str = "", xlabel: str = "", ylabel: str = "") -> str:
    """Multiple line series sharing an x-axis (e.g. static vs adaptive over time).

    ``series`` maps a label -> list of y-values (same length as x).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, ys in series.items():
        ax.plot(x, ys, "o-", label=label, linewidth=1.8, markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return save_fig(fig, output_dir, name)
