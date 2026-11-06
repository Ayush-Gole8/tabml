"""Generate the HPO cost-vs-benefit figure for the IEEE Access paper.

Reads the per-dataset results.json produced by TabML-Guard and plots, for each
dataset, the *relative improvement of the held-out test score over the untuned
default* against the wall-clock seconds the search consumed.  Categorical colour
encodes the dataset (3 slots, all-pairs validated); marker shape encodes the
search strategy, so identity is never carried by colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "cost_benefit.png"

# --- validated reference palette (light surface) -----------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DATASET_COLOR = {
    "electricity": "#2a78d6",      # slot 1 blue
    "appliances-energy": "#eb6834",  # slot 2 orange
    "metro-traffic": "#1baf7a",    # slot 3 aqua
}
DATASET_LABEL = {
    "electricity": "D1  ELEC2 (ROC-AUC)",
    "appliances-energy": "D2  Appliances (RMSE)",
    "metro-traffic": "D3  Metro traffic (RMSE)",
}
STRATEGY_MARKER = {"default": "o", "grid": "s", "random": "^", "bayes": "D"}
STRATEGY_ORDER = ["default", "grid", "random", "bayes"]


def load(ds: str):
    """Return [(strategy, wall_seconds, relative_gain_pct), ...]."""
    d = json.loads((ROOT / "results" / ds / "tables" / "results.json").read_text())
    tuning = d["tuning"]
    booster = tuning["booster"]
    leaderboard = {r["name"]: r["primary"] for r in tuning["leaderboard"]}

    def test_score(strategy: str) -> float:
        name = booster if strategy == "default" else f"{booster}_{strategy}"
        return leaderboard.get(name, leaderboard.get(f"{booster}_default"))

    scores = {r["strategy"]: (r["wall_seconds"], test_score(r["strategy"]))
              for r in tuning["records"]}
    base = scores["default"][1]
    # RMSE is minimised, ROC-AUC maximised -> express both as "higher is better".
    lower_is_better = tuning["primary_metric"] == "rmse"
    out = []
    for s in STRATEGY_ORDER:
        wall, score = scores[s]
        gain = (base - score) / base if lower_is_better else (score - base) / base
        out.append((s, wall, gain * 100.0))
    return out


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.15), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    series = {ds: load(ds) for ds in DATASET_COLOR}

    # recessive chrome
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)

    for ds, pts in series.items():
        color = DATASET_COLOR[ds]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, "-", color=color, linewidth=1.6, zorder=2, alpha=0.85)
        for strategy, x, y in pts:
            ax.scatter(
                x, y, s=95, marker=STRATEGY_MARKER[strategy], color=color,
                edgecolors=SURFACE, linewidths=1.6, zorder=4,
            )
        # selective direct label: the endpoint of each cost curve
        last_x, last_y = xs[-1], ys[-1]
        ax.annotate(
            f"{last_y:+.1f}%", xy=(last_x, last_y), xytext=(9, 0),
            textcoords="offset points", va="center", ha="left",
            fontsize=8, color=INK_2, zorder=5,
        )

    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 2600)
    ax.set_ylim(-3.2, 22)

    ax.set_xlabel("Hyperparameter-search wall-clock cost (s, log scale)",
                  fontsize=8.5, color=INK_2)
    ax.set_ylabel("Gain over untuned default\n(%, higher = better)",
                  fontsize=8.5, color=INK_2)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)

    # the zero line is the whole point: everything below it is tuning that hurt
    ax.annotate("below the line: tuning made the model worse",
                xy=(3.0, -1.35), fontsize=7.4, color=MUTED, style="italic")

    legend_ds = [
        Line2D([], [], color=DATASET_COLOR[ds], marker="o", linestyle="-",
               linewidth=1.6, markersize=6, markeredgecolor=SURFACE,
               markeredgewidth=1.2, label=DATASET_LABEL[ds])
        for ds in series
    ]
    legend_strategy = [
        Line2D([], [], color=MUTED, marker=STRATEGY_MARKER[s], linestyle="none",
               markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
               label=s.capitalize())
        for s in STRATEGY_ORDER
    ]
    first = ax.legend(handles=legend_ds, loc="upper left", fontsize=7.6,
                      frameon=False, labelcolor=INK_2, handletextpad=0.6,
                      borderaxespad=0.4)
    ax.add_artist(first)
    ax.legend(handles=legend_strategy, loc="lower right", fontsize=7.6,
              frameon=False, labelcolor=INK_2, handletextpad=0.6,
              ncol=2, columnspacing=1.1, borderaxespad=0.4)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print("wrote", OUT)
    for ds, pts in series.items():
        print(ds, [(s, w, round(g, 3)) for s, w, g in pts])


if __name__ == "__main__":
    main()
