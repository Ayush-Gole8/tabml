"""TabML-Guard entry point.

Usage:
    python run.py --config config.yaml

Reads the config, runs the pipeline stage by stage, and writes a professional PDF
report. All stages are live: Stage 0 (data/profiling/temporal split), Stage 1
(regularized baselines), Stage 2 (boosting + stacking), Stage 3 (HPO comparison),
Stage 4 (SHAP + reliability), Stage 5 (concept-drift monitoring + significance tests).

Every run writes into its own per-dataset directory so results from different
configs never overwrite one another::

    results/<dataset>/
        report_<dataset>.pdf     <- the deliverable
        best_model.pkl
        figures/*.png
        tables/{profile,results}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tabml_guard import baselines, boosting, drift, explain, paths, pdf_report, report, tuning
from tabml_guard.config import load_config
from tabml_guard.data import prepare


def _banner(msg: str) -> None:
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def _save_json(obj: dict, cfg, name: str) -> None:
    """Write a JSON-safe dict into this run's tables/ directory."""
    (paths.tables_dir(cfg) / f"{name}.json").write_text(
        json.dumps(obj, indent=2), encoding="utf-8")


def _print_leaderboard(stage_summary: dict, primary_metric: str) -> None:
    """Print the merged best-first leaderboard a stage attached to its summary."""
    lb = stage_summary.get("leaderboard") if isinstance(stage_summary, dict) else None
    if not lb:
        return
    print(f"\n[leaderboard] ranked by {primary_metric} (all models so far):")
    for i, row in enumerate(lb, 1):
        v = row.get("primary", float("nan"))
        print(f"   {i:>2}. {row['name']:<20} {primary_metric}={v:.4f}  ({row['fit_seconds']}s)")


def _print_significance(drift_summary: dict) -> None:
    """Print the McNemar/Wilcoxon head-to-heads a drift summary attached."""
    sig = drift_summary.get("significance") if isinstance(drift_summary, dict) else None
    if not sig:
        return
    print("\n[significance] statistical tests on key claims:")
    for label, r in sig.items():
        p = r.get("p_value")
        mark = "SIGNIFICANT" if r.get("significant") else "not significant"
        p_txt = "n/a" if p is None else f"{p:.4g}"
        print(f"   - {label}: {r.get('better_model')} better  (p={p_txt}, {mark})")


def _save_best_model(ctx: dict, cfg) -> None:
    """Persist the best fitted pipeline so it can be reused/deployed."""
    best = ctx.get("best_overall")
    if best is None:
        return
    try:
        import joblib

        path = paths.model_path(cfg)
        joblib.dump(best.estimator, path)
        print(f"[model] saved best model '{best.name}' -> {path}")
    except Exception as exc:
        print(f"[model] could not save best model: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TabML-Guard pipeline")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _banner(f"TabML-Guard  |  {cfg.report.title}")
    run_out = paths.run_dir(cfg)
    print(f"[paths] this run writes into: {run_out}")

    # ----- Stage 0: data -----------------------------------------------------
    _banner("Stage 0 — Data ingestion, profiling & temporal split")
    bundle = prepare(cfg)
    _save_json(bundle.profile, cfg, "profile")

    prof = bundle.profile
    print(f"\n[summary] rows={prof['n_rows']:,}  features={prof['n_features']}  task={bundle.task}")
    if bundle.task == "classification":
        print(f"[summary] classes={bundle.target_classes}  balance={prof.get('class_balance')}")
    else:
        print(f"[summary] target stats={prof.get('target_stats')}")
    if prof.get("missing_total", 0):
        print(f"[summary] missing values: {prof['missing_total']:,}")
    print(f"[summary] time-ordered: {bundle.has_time}")

    # ----- Stages 1-5 --------------------------------------------------------
    # `ctx` is an in-memory scratchpad shared across stages: it carries fitted
    # models, predictions and the running "best model" between stages. It is NOT
    # serialized. `report_ctx` holds only JSON-safe per-stage summaries — that is
    # what gets saved and handed to the report.
    ctx: dict = {}
    report_ctx: dict = {"profile": bundle.profile, "task": bundle.task}
    primary_metric = "roc_auc" if bundle.task == "classification" else "rmse"

    _banner("Stage 1 — Regularized baselines")
    report_ctx["baselines"] = baselines.run(bundle, cfg, ctx)
    _print_leaderboard(report_ctx["baselines"], primary_metric)
    if report_ctx["baselines"].get("note"):
        print(f"[stage1] {report_ctx['baselines']['note']}")

    _banner("Stage 2 — Ensemble / boosting models")
    report_ctx["boosting"] = boosting.run(bundle, cfg, ctx)
    _print_leaderboard(report_ctx["boosting"], primary_metric)
    if ctx.get("best_overall") is not None:
        best = ctx["best_overall"]
        print(
            f"\n[pipeline] best model overall: '{best.name}' "
            f"({primary_metric}={best.metrics.get(primary_metric, float('nan')):.4f}). "
            f"Stage 3 will tune the best booster: '{ctx.get('best_booster_name')}'."
        )

    _banner("Stage 3 — Hyperparameter-optimization comparison")
    report_ctx["tuning"] = tuning.run(bundle, cfg, ctx)
    _print_leaderboard(report_ctx["tuning"], primary_metric)
    if report_ctx["tuning"].get("verdict"):
        print(f"\n[stage3] verdict: {report_ctx['tuning']['verdict']}")

    _banner("Stage 4 — SHAP explainability + reliability")
    report_ctx["explain"] = explain.run(bundle, cfg, ctx)  # prints its own summary lines

    _banner("Stage 5 — Concept-drift monitoring + significance tests")
    report_ctx["drift"] = drift.run(bundle, cfg, ctx)       # prints its own verdict
    _print_significance(report_ctx["drift"])

    # Persist the best fitted model + a JSON-safe run summary.
    _save_best_model(ctx, cfg)
    _save_json(report_ctx, cfg, "results")

    # ----- Report ------------------------------------------------------------
    _banner("Building Reports (PDF & HTML)")
    pdf_path = pdf_report.build(report_ctx, str(run_out), cfg.report.title, cfg=cfg)
    html_path = report.build(report_ctx, str(run_out), cfg.report.title, cfg=cfg)
    print(f"[report] PDF:  {pdf_path}")
    print(f"[report] HTML: {html_path}")

    _banner("Done")
    print("Pipeline complete: baselines -> ensembles -> HPO -> SHAP/reliability -> drift/significance.")
    print(f"Interactive HTML: {html_path}")
    print(f"Document PDF:     {pdf_path}")
    print(f"All artifacts for this dataset: {run_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
