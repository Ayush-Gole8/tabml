"""Build results/INDEX.md — a cross-config comparison of every run.

Each run leaves a self-contained folder under ``results/<dataset>/``. This script
scans those folders' ``tables/results.json`` files and emits one Markdown table
summarising all of them side by side, so you can compare datasets at a glance
without opening five reports.

Usage:
    python make_index.py                 # scans ./results
    python make_index.py --results-dir out
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _num(v: Any, nd: int = 4) -> str:
    if v is None or _is_nan(v):
        return "-"
    if isinstance(v, int):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) >= 1000:
        return f"{int(f):,}"
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 0) -> str:
    if v is None or _is_nan(v):
        return "-"
    return f"{float(v) * 100:.{nd}f}%"


def _leaderboard(ctx: dict) -> list:
    for stage in ("tuning", "boosting", "baselines"):
        lb = (ctx.get(stage) or {}).get("leaderboard")
        if lb:
            return lb
    return []


def _drift_label(ctx: dict) -> str:
    """One-word drift summary, derived the same way the report does it."""
    dr = ctx.get("drift") or {}
    if dr.get("status") != "ok":
        return dr.get("status", "n/a")
    trend = dr.get("trend") or {}
    fd = dr.get("feature_drift") or {}
    psi = fd.get("max_mean_psi")
    moved = psi is not None and not _is_nan(psi) and float(psi) > 0.25
    worsened = bool(trend.get("worsened"))
    if worsened and moved:
        return "drift: inputs + accuracy"
    if moved:
        return "inputs drifted, held"
    if worsened:
        return "accuracy decayed"
    return "stable"


def summarise(run_dir: Path) -> Optional[dict]:
    """Pull the headline facts out of one run directory."""
    jf = run_dir / "tables" / "results.json"
    if not jf.exists():
        return None
    try:
        ctx = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        return None

    prof = ctx.get("profile", {}) or {}
    task = ctx.get("task", "-")
    primary = (ctx.get("baselines") or {}).get("primary_metric") \
        or ("roc_auc" if task == "classification" else "rmse")
    lb = _leaderboard(ctx)
    champ = lb[0].get("name", "-") if lb else "-"
    champ_score = lb[0].get("primary") if lb else None

    tun = ctx.get("tuning") or {}
    ex = ctx.get("explain") or {}
    flag = (ex.get("reliability") or {}).get("flag_rule") or {}

    reports = sorted(p.name for p in run_dir.glob("report_*.pdf"))
    return {
        "dataset": run_dir.name,
        "task": task,
        "rows": prof.get("n_rows"),
        "features": prof.get("n_features"),
        "target": prof.get("target", "-"),
        "time_col": prof.get("time_col") or "none",
        "primary": primary,
        "champion": champ,
        "score": champ_score,
        "best_strategy": tun.get("best_strategy", "-"),
        "drift": _drift_label(ctx),
        "error_capture": flag.get("error_capture_rate"),
        "report": reports[0] if reports else "",
    }


def build_index(results_dir: Path) -> str:
    runs = []
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        s = summarise(d)
        if s:
            runs.append(s)

    lines = [
        "# Results index",
        "",
        f"{len(runs)} run(s) found in `{results_dir.as_posix()}/`. "
        "Each row is one config; every dataset keeps its own folder with its "
        "report, figures, tables and saved model.",
        "",
    ]
    if not runs:
        lines += [
            "_No runs yet._ Run the pipeline to populate this index:",
            "",
            "```bash",
            "python run.py --config config.yaml",
            "python make_index.py",
            "```",
            "",
        ]
        return "\n".join(lines)

    # --- headline comparison table ---
    lines += [
        "## All runs",
        "",
        "| Dataset | Task | Rows | Features | Champion model | Metric | Score | Best HPO | Drift verdict |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| [`{r['dataset']}`]({r['dataset']}/) | {r['task']} | {_num(r['rows'])} | "
            f"{_num(r['features'])} | `{r['champion']}` | {r['primary']} | "
            f"{_num(r['score'])} | {r['best_strategy']} | {r['drift']} |"
        )
    lines.append("")

    # --- per-run detail ---
    lines += ["## Per-run detail", ""]
    for r in runs:
        lines.append(f"### `{r['dataset']}`")
        lines.append("")
        lines.append(f"- **Target / time column** — `{r['target']}` ordered by `{r['time_col']}`")
        lines.append(f"- **Champion** — `{r['champion']}` at {r['primary']} = "
                     f"`{_num(r['score'])}` on the held-out period")
        lines.append(f"- **Best tuning strategy** — {r['best_strategy']}")
        lines.append(f"- **Drift** — {r['drift']}")
        if r["error_capture"] is not None and not _is_nan(r["error_capture"]):
            lines.append(f"- **Confidence triage** — flagging the least-confident slice "
                         f"catches {_pct(r['error_capture'])} of all errors")
        if r["report"]:
            lines.append(f"- **Report** — [`{r['report']}`]({r['dataset']}/{r['report']})")
        lines.append(f"- **Artifacts** — `{r['dataset']}/figures/`, "
                     f"`{r['dataset']}/tables/`, `{r['dataset']}/best_model.pkl`")
        lines.append("")

    lines += [
        "---",
        "",
        "Regenerate with `python make_index.py`.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build results/INDEX.md")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)

    rd = Path(args.results_dir)
    if not rd.exists():
        print(f"[index] no results directory at {rd} - nothing to do.")
        return 1
    out = rd / "INDEX.md"
    out.write_text(build_index(rd), encoding="utf-8")
    print(f"[index] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
