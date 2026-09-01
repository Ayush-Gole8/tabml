"""Stage 6 — the headline deliverable: a self-contained HTML report.

Turns the JSON-safe ``report_ctx`` produced by a run into a single, standalone
``report.html`` that a non-expert can read top-to-bottom and walk away knowing:

  * which model to use and how good it is (the verdict, up top),
  * why it decides what it decides (SHAP), when to trust it (reliability),
  * whether it will keep working as time passes (drift), and
  * whether every "X beats Y" claim is statistically real (significance),
  * **and how to run this same pipeline on their own tabular dataset.**

Design contract
---------------
* **Self-contained.** No web fonts, no CDN, no JavaScript. Figures are read from
  ``<output_dir>/figures/*.png`` and inlined as base64 data URIs, and the two
  chart-less diagnostics (regularization path, learning curve) are drawn as inline
  SVG. The file opens anywhere, offline, forever.
* **Dataset-agnostic.** Every sentence and number is derived from the context, so
  the report is correct whether the run was classification or regression, whether
  drift/tuning ran or were skipped, and whatever dataset was plugged in.
* **Never raises.** Missing/skipped/errored stages degrade to a clear note rather
  than crashing the report.

Only the Python standard library is used here (base64, html, datetime, pathlib).
"""
from __future__ import annotations

import base64
import datetime
import html as _html
import math
import re
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Metric metadata (task-agnostic rendering)
# --------------------------------------------------------------------------- #
_CLF_METRICS = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "f2"]
_REG_METRICS = ["rmse", "mae", "r2"]
_METRIC_LABEL = {
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "accuracy": "Accuracy",
    "precision": "Precision", "recall": "Recall", "f1": "F1", "f2": "F₂",
    "rmse": "RMSE", "mae": "MAE", "r2": "R²",
}
_LOWER_BETTER = {"rmse", "mae"}


# --------------------------------------------------------------------------- #
# Small formatting / HTML helpers
# --------------------------------------------------------------------------- #
def esc(x: Any) -> str:
    return _html.escape(str(x), quote=True)


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _num(v: Any, nd: int = 4) -> str:
    """Format a number for the instrument readout; em-dash for missing/NaN."""
    if v is None or _is_nan(v):
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    if f == int(f) and abs(f) >= 1000:
        return f"{int(f):,}"
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 1) -> str:
    if v is None or _is_nan(v):
        return "—"
    return f"{float(v) * 100:.{nd}f}%"


def _secs(v: Any) -> str:
    if v is None or _is_nan(v):
        return "—"
    return f"{float(v):.1f}s"


def _mono(s: str) -> str:
    return f'<span class="mono">{s}</span>'


def _pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="pill pill--{kind}">{esc(text)}</span>'


def _callout(title: str, body_html: str, kind: str = "note") -> str:
    return (
        f'<div class="callout callout--{kind}">'
        f'<div class="callout__title">{esc(title)}</div>'
        f'<div class="callout__body">{body_html}</div></div>'
    )


def _table(headers: list[str], rows: list[list[str]], *, highlight_row: int = -1,
           right_from: int = 1) -> str:
    """Build a table. Cells at/after column ``right_from`` are right-aligned mono."""
    head = "".join(
        f'<th class="{"num" if i >= right_from else ""}">{h}</th>'
        for i, h in enumerate(headers)
    )
    body_rows = []
    for r_i, row in enumerate(rows):
        cls = ' class="row--best"' if r_i == highlight_row else ""
        cells = "".join(
            f'<td class="{"num mono" if i >= right_from else ""}">{c}</td>'
            for i, c in enumerate(row)
        )
        body_rows.append(f"<tr{cls}>{cells}</tr>")
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def _embed_figure(output_dir: str, name: str, caption: str) -> str:
    """Read <output_dir>/figures/<name>.png and inline it; '' if absent."""
    path = Path(output_dir) / "figures" / f"{name}.png"
    if not path.exists():
        return ""
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return (
        f'<figure class="fig"><img alt="{esc(caption)}" '
        f'src="data:image/png;base64,{data}"/>'
        f'<figcaption>{esc(caption)}</figcaption></figure>'
    )


def _svg_line(series: dict[str, list[Optional[float]]], *, title: str = "",
              ylabel: str = "", x_first: str = "", x_last: str = "",
              width: int = 620, height: int = 210) -> str:
    """A compact inline-SVG multi-line chart (no matplotlib needed)."""
    vals = [v for ys in series.values() for v in ys if v is not None and not _is_nan(v)]
    if len(vals) < 2:
        return ""
    ymin, ymax = min(vals), max(vals)
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    pad_l, pad_t, pad_b, pad_r = 48, 30, 30, 14
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = max(max(len(ys) for ys in series.values()), 2)

    def X(i: int) -> float:
        return pad_l + plot_w * (i / (n - 1))

    def Y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - ymin) / (ymax - ymin))

    colors = ["#0f766e", "#14213d", "#b45309", "#b91c1c"]
    parts = [f'<svg class="svgchart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{esc(title)}">']
    if title:
        parts.append(f'<text x="{pad_l}" y="16" class="svg-title">{esc(title)}</text>')
    # horizontal gridlines + y labels (max / mid / min)
    for frac, val in [(0.0, ymax), (0.5, (ymin + ymax) / 2), (1.0, ymin)]:
        y = pad_t + plot_h * frac
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                     f'y2="{y:.1f}" class="svg-grid"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 3:.1f}" class="svg-ytick">'
                     f'{val:.3f}</text>')
    # x endpoints
    if x_first:
        parts.append(f'<text x="{pad_l}" y="{height - 8}" class="svg-xtick">{esc(x_first)}</text>')
    if x_last:
        parts.append(f'<text x="{width - pad_r}" y="{height - 8}" '
                     f'class="svg-xtick svg-xtick--end">{esc(x_last)}</text>')
    # series
    legend = []
    for idx, (label, ys) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        pts = " ".join(f"{X(j):.1f},{Y(v):.1f}" for j, v in enumerate(ys)
                       if v is not None and not _is_nan(v))
        if pts:
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                         f'stroke-width="2"/>')
            for j, v in enumerate(ys):
                if v is not None and not _is_nan(v):
                    parts.append(f'<circle cx="{X(j):.1f}" cy="{Y(v):.1f}" r="2.4" '
                                 f'fill="{color}"/>')
        legend.append((label, color))
    lx = pad_l
    for label, color in legend:
        parts.append(f'<rect x="{lx}" y="{pad_t - 12}" width="10" height="10" '
                     f'rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{pad_t - 3}" class="svg-legend">'
                     f'{esc(label)}</text>')
        lx += 22 + 7 * len(label)
    if ylabel:
        parts.append(f'<text x="12" y="{pad_t + plot_h / 2}" class="svg-ylabel" '
                     f'transform="rotate(-90 12 {pad_t + plot_h / 2})">{esc(ylabel)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Leaderboard / metric rendering
# --------------------------------------------------------------------------- #
def _metric_keys(task: str) -> list[str]:
    return _CLF_METRICS if task == "classification" else _REG_METRICS


def _leaderboard_table(lb: list[dict], task: str, primary: str) -> str:
    if not lb:
        return '<p class="muted">No models to rank.</p>'
    keys = _metric_keys(task)
    # primary first, then the rest
    ordered = [primary] + [k for k in keys if k != primary]
    arrow = "↓" if primary in _LOWER_BETTER else "↑"
    headers = ["#", "Model"] + [
        (f"{_METRIC_LABEL.get(k, k)} {arrow}" if k == primary else _METRIC_LABEL.get(k, k))
        for k in ordered
    ] + ["Fit"]
    rows = []
    for i, row in enumerate(lb, 1):
        m = row.get("metrics", {})
        cells = [str(i), esc(row.get("name", "?"))]
        for k in ordered:
            cells.append(_num(m.get(k)))
        cells.append(_secs(row.get("fit_seconds")))
        rows.append(cells)
    return _table(headers, rows, highlight_row=0, right_from=2)


# --------------------------------------------------------------------------- #
# Insight engine (drives the executive summary)
# --------------------------------------------------------------------------- #
def _merged_leaderboard(ctx: dict) -> list[dict]:
    for stage in ("tuning", "boosting", "baselines"):
        lb = (ctx.get(stage) or {}).get("leaderboard")
        if lb:
            return lb
    return []


def _score_of(lb: list[dict], name: str) -> Optional[float]:
    for r in lb:
        if r.get("name") == name:
            return r.get("primary")
    return None


def _family(name: str, ctx: dict) -> str:
    if not name:
        return "model"
    base_names = {m["name"] for m in (ctx.get("baselines") or {}).get("models", [])}
    if name in base_names:
        return "regularized linear model"
    if name == "stacking":
        return "stacking ensemble"
    for suf in ("_default", "_grid", "_random", "_bayes"):
        if name.endswith(suf):
            return f"tuned {name[: -len(suf)]}"
    return f"{name} ensemble"


def _better(a: Optional[float], b: Optional[float], primary: str) -> bool:
    """Is score a better than score b under the primary metric's direction?"""
    if a is None or b is None or _is_nan(a) or _is_nan(b):
        return False
    return a < b if primary in _LOWER_BETTER else a > b


def _insights(ctx: dict, task: str, primary: str) -> dict:
    """Compute champion, status pills, narrative and recommendation from context."""
    lb = _merged_leaderboard(ctx)
    champ = lb[0]["name"] if lb else None
    champ_score = lb[0]["primary"] if lb else None
    fam = _family(champ, ctx)

    pills: list[tuple[str, str]] = [(f"Champion: {fam}", "accent")]

    # Simple vs complex.
    best_base = (ctx.get("baselines") or {}).get("best")
    best_booster = (ctx.get("boosting") or {}).get("best_booster")
    base_names = {m["name"] for m in (ctx.get("baselines") or {}).get("models", [])}
    complexity_note = ""
    if champ in base_names and best_booster:
        pills.append(("Simple model wins", "caution"))
        complexity_note = (
            "A simple regularized linear model matched or beat the gradient-boosted "
            "ensembles here — added model complexity did not pay off on this dataset."
        )
    elif champ == "stacking" or (best_booster and champ and champ.startswith(best_booster)):
        pills.append(("Ensemble wins", "good"))
        complexity_note = "A boosted / stacked ensemble was the top performer."

    # Tuning verdict.
    tuning = ctx.get("tuning") or {}
    if tuning.get("status") == "ok" and tuning.get("records"):
        tuned_best = tuning.get("tuned_best")
        default_score = _score_of(lb, f"{tuning.get('booster')}_default")
        tuned_score = _score_of(lb, tuned_best) if tuned_best else None
        if default_score is not None and tuned_score is not None:
            gain = (default_score - tuned_score) if primary in _LOWER_BETTER else (tuned_score - default_score)
            if gain > 0.005:
                pills.append(("Tuning helped", "good"))
            elif gain > 0:
                pills.append(("Tuning: marginal gain", "caution"))
            else:
                pills.append(("Tuning: no gain", "neutral"))

    # Drift risk.
    drift = ctx.get("drift") or {}
    if drift.get("status") == "skipped":
        pills.append(("Drift: not assessed", "neutral"))
    elif drift.get("status") == "ok":
        trend = drift.get("trend", {})
        max_psi = (drift.get("feature_drift") or {}).get("max_mean_psi", 0) or 0
        if trend.get("worse_over_time") and abs(trend.get("drift_magnitude", 0)) > 0.02:
            pills.append(("Drift: performance decayed", "alert"))
        elif max_psi > 0.2:
            pills.append(("Inputs drifted, performance held", "caution"))
        else:
            pills.append(("Stable over time", "good"))

    # Trust / triage (classification reliability only).
    rel = (ctx.get("explain") or {}).get("reliability")
    if rel and rel.get("error_capture_rate") is not None:
        pills.append((f"Triage flag catches {_pct(rel['error_capture_rate'], 0)} of errors", "accent"))

    # ---- Recommendation ----
    rec_bits = []
    if champ:
        interp = " It is also fast and directly interpretable." if champ in base_names else ""
        rec_bits.append(
            f"Deploy <b>{esc(champ)}</b> ({fam}); it leads the leaderboard at "
            f"{_mono(_num(champ_score))} {esc(_METRIC_LABEL.get(primary, primary))}.{interp}"
        )
    if drift.get("status") == "ok":
        sig = drift.get("significance", {}) or {}
        sva = sig.get("static_vs_adaptive", {})
        adaptive_mean = drift.get("adaptive_mean")
        if adaptive_mean is None:
            rec_bits.append("Retraining was not evaluated (retrain mode = none).")
        elif sva and not sva.get("significant"):
            rec_bits.append(
                f"Periodic retraining ({esc(drift.get('retrain_mode'))}) is <b>not</b> currently "
                "justified — it did not significantly beat the frozen model — but keep "
                "monitoring input drift (PSI) and revisit if it climbs."
            )
        elif sva and sva.get("significant"):
            rec_bits.append(
                f"Adopt {esc(drift.get('retrain_mode'))} retraining — it significantly "
                "outperformed the frozen model over time."
            )
    if rel and rel.get("error_capture_rate") is not None:
        rec_bits.append(
            f"Route the least-confident {_pct(rel.get('coverage_flagged'), 0)} of predictions "
            f"to human review to catch {_pct(rel.get('error_capture_rate'), 0)} of all errors."
        )

    return {
        "champion": champ,
        "champion_score": champ_score,
        "family": fam,
        "pills": pills,
        "complexity_note": complexity_note,
        "recommendation": " ".join(rec_bits),
        "leaderboard": lb,
    }


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _skipped_note(stage: dict, label: str) -> str:
    status = stage.get("status", "pending")
    reason = stage.get("reason", "not run")
    return _callout(f"{label}: {status}", f"<p>{esc(reason)}.</p>", "neutral")


def _section(sid: str, eyebrow: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="section__sub">{esc(subtitle)}</p>' if subtitle else ""
    return (
        f'<section id="{sid}" class="section">'
        f'<div class="section__head"><span class="eyebrow">{esc(eyebrow)}</span>'
        f'<h2>{esc(title)}</h2>{sub}</div>{body}</section>'
    )


def _sec_dataset(ctx: dict, cfg_block: str) -> str:
    p = ctx.get("profile", {})
    task = ctx.get("task", "?")
    rows = [
        ["Rows", _num(p.get("n_rows"))],
        ["Features", _num(p.get("n_features"))],
        ["Task", esc(task)],
        ["Target column", esc(p.get("target", "—"))],
        ["Missing values", _num(p.get("missing_total"))],
    ]
    if p.get("time_col"):
        span = p.get("time_span")
        rows.append(["Time column", esc(p["time_col"])])
        if span:
            rows.append(["Time span", f"{esc(span[0])} → {esc(span[1])}"])
    kv = _table(["Property", "Value"], rows, right_from=1)

    dist = ""
    if task == "classification" and p.get("class_balance"):
        bars = []
        for cls, frac in p["class_balance"].items():
            cnt = (p.get("class_counts") or {}).get(cls)
            bars.append(
                f'<div class="bar"><div class="bar__label">{esc(cls)}</div>'
                f'<div class="bar__track"><div class="bar__fill" style="width:{float(frac)*100:.1f}%"></div></div>'
                f'<div class="bar__val mono">{_pct(frac)} ({_num(cnt)})</div></div>'
            )
        dist = ('<h3>Class balance</h3><div class="bars">' + "".join(bars) + "</div>")
    elif task == "regression" and p.get("target_stats"):
        ts = p["target_stats"]
        dist = "<h3>Target statistics</h3>" + _table(
            ["Stat", "Value"],
            [["min", _num(ts.get("min"))], ["max", _num(ts.get("max"))],
             ["mean", _num(ts.get("mean"))], ["std", _num(ts.get("std"))]],
            right_from=1,
        )

    intro = (
        "<p>This is the data the pipeline learned from. For time-ordered data the split is "
        "<b>chronological</b> — the model trains on the earliest rows and is tested on the "
        "latest, which is the honest way to estimate how it will behave on future data.</p>"
    )
    return _section("s0", "S0 · Data", "Dataset profile & split",
                    intro + '<div class="grid2">' + kv + dist + "</div>" + cfg_block)


def _sec_baselines(ctx: dict, task: str, primary: str) -> str:
    b = ctx.get("baselines", {})
    if b.get("status") != "ok":
        return _section("s1", "S1 · Baselines", "Regularized baselines",
                        _skipped_note(b, "Baselines"))
    intro = (
        "<p>Before reaching for heavy models we fit simple, <b>regularized</b> linear models. "
        "Regularization is a penalty that discourages over-complex fits: <b>L1</b> can switch "
        "features off entirely (feature selection), <b>L2</b> gently shrinks them, and "
        "<b>ElasticNet</b> blends both. These set the bar every fancier model must clear.</p>"
    )
    body = intro + _leaderboard_table(b.get("leaderboard", []), task, primary)
    if b.get("note"):
        body += _callout("What the numbers say", f"<p>{esc(b['note'])}</p>", "note")

    # Regularization path (inline SVG) — the pedagogical payload.
    path = b.get("regularization_path")
    if path and path.get("C") and path.get("l1"):
        xs = path["C"]
        l1 = path["l1"]
        l2 = path.get("l2", {})
        chart = _svg_line(
            {"L1 (Lasso-style)": l1.get("val_" + primary) or l1.get("val_roc_auc"),
             "L2 (Ridge-style)": (l2.get("val_" + primary) or l2.get("val_roc_auc")) if l2 else []},
            title="Validation score vs regularization strength (C: weak → strong penalty is right → left)",
            ylabel=_METRIC_LABEL.get(primary, primary),
            x_first=f"C={xs[0]:g}", x_last=f"C={xs[-1]:g}",
        )
        spark = _svg_line(
            {"# features kept (L1)": [float(v) for v in l1.get("nonzero", [])]},
            title=f"L1 feature selection: non-zero coefficients out of {path.get('n_features', '?')}",
            ylabel="features", x_first=f"C={xs[0]:g}", x_last=f"C={xs[-1]:g}",
        )
        body += ('<h3>How regularization reshapes the model</h3>'
                 '<p class="muted">Left = strong penalty (simpler model); right = weak penalty '
                 '(complex model). L1 drives coefficients to exactly zero, so it doubles as '
                 'automatic feature selection.</p>' + chart + spark)
    return _section("s1", "S1 · Baselines", "Regularized baselines", body)


def _sec_boosting(ctx: dict, task: str, primary: str) -> str:
    bo = ctx.get("boosting", {})
    if bo.get("status") != "ok":
        return _section("s2", "S2 · Ensembles", "Boosting & stacking",
                        _skipped_note(bo, "Ensembles"))
    intro = (
        "<p>Next, the workhorses: gradient-boosting libraries (XGBoost, LightGBM, CatBoost) that "
        "build many small decision trees in sequence, a Random Forest as an older-ensemble "
        "reference, and a <b>stacking</b> model that lets a regularized linear model combine all "
        "of them. Any library you haven't installed is skipped automatically.</p>"
    )
    body = intro + _leaderboard_table(bo.get("leaderboard", []), task, primary)
    avail = ", ".join(bo.get("available_boosters", [])) or "none"
    skipped = ", ".join(bo.get("skipped_boosters", [])) or "none"
    body += _callout(
        "Models trained",
        f"<p>Available: {_mono(esc(avail))}. Skipped (not installed): {_mono(esc(skipped))}. "
        f"Best single booster: <b>{esc(bo.get('best_booster', '—'))}</b> — this is what "
        f"Stage 3 tunes.</p>", "note")

    # Top feature importances of the best ensemble.
    best_name = bo.get("best")
    imp = None
    for m in bo.get("models", []):
        if m.get("name") == best_name:
            imp = (m.get("extra") or {}).get("importance", {}).get("top_features")
    if imp:
        top = imp[:10]
        vmax = max((v for _, v in top), default=1) or 1
        bars = "".join(
            f'<div class="bar"><div class="bar__label">{esc(f)}</div>'
            f'<div class="bar__track"><div class="bar__fill" style="width:{(v / vmax) * 100:.1f}%"></div></div>'
            f'<div class="bar__val mono">{_num(v, 4)}</div></div>' for f, v in top
        )
        body += (f"<h3>What drives <span class='mono'>{esc(best_name)}</span></h3>"
                 f'<div class="bars">{bars}</div>')
    return _section("s2", "S2 · Ensembles", "Boosting & stacking", body)


def _sec_tuning(ctx: dict, task: str, primary: str) -> str:
    t = ctx.get("tuning", {})
    if t.get("status") != "ok":
        return _section("s3", "S3 · Tuning", "Hyperparameter optimization",
                        _skipped_note(t, "Tuning"))
    intro = (
        f"<p>Hyperparameters are the knobs you set <i>before</i> training (tree depth, learning "
        f"rate, ...). We tune <b>{esc(t.get('booster', '?'))}</b> four ways and, crucially, record "
        f"the <b>compute cost</b> of each so we can answer the real question: <i>is the smarter "
        f"search worth the extra time?</i> Grid and Random and Bayesian all search the same space; "
        f"Random and Bayesian get the <b>same budget</b> ({_num(t.get('budget_trials'))} trials) "
        f"for a fair head-to-head.</p>"
    )
    # Cost/benefit table from records.
    recs = t.get("records", [])
    lb = t.get("leaderboard", [])
    rows = []
    for r in recs:
        strat = r.get("strategy", "?")
        model_name = f"{t.get('booster')}_{strat}"
        rows.append([
            esc(strat),
            _num(r.get("cv_score")),
            _num(_score_of(lb, model_name)),
            _num(r.get("n_evaluations")),
            _secs(r.get("wall_seconds")),
        ])
    cost = _table(
        ["Strategy", "CV score", f"Test {_METRIC_LABEL.get(primary, primary)}",
         "# fits", "Wall time"],
        rows, right_from=1)
    body = intro + cost
    if t.get("verdict"):
        body += _callout("Is the expensive search worth it?", f"<p>{esc(t['verdict'])}</p>", "note")
    body += _leaderboard_table(lb, task, primary)
    return _section("s3", "S3 · Tuning", "Hyperparameter optimization", body)


def _sec_explain(ctx: dict, output_dir: str) -> str:
    e = ctx.get("explain", {})
    if e.get("status") != "ok":
        return _section("s4", "S4 · Explain", "Explainability & reliability",
                        _skipped_note(e, "Explainability"))
    intro = (
        f"<p>Two trust questions about the champion "
        f"(<span class='mono'>{esc(e.get('model', '?'))}</span>): <b>why</b> does it decide what it "
        f"decides, and <b>when</b> should a human double-check it?</p>"
    )
    body = intro

    shap = e.get("shap")
    if shap and shap.get("global_importance"):
        body += (f"<h3>Why: feature importance (SHAP, {esc(shap.get('method'))})</h3>"
                 "<p class='muted'>SHAP splits each prediction into the contribution of every "
                 "input. Averaged over many rows, it shows which inputs matter most.</p>")
        fig = _embed_figure(output_dir, "shap_importance", "Global feature importance (mean |SHAP|)")
        if fig:
            body += fig
        else:  # fall back to a bar list
            top = shap["global_importance"][:10]
            vmax = max((v for _, v in top), default=1) or 1
            bars = "".join(
                f'<div class="bar"><div class="bar__label">{esc(f)}</div>'
                f'<div class="bar__track"><div class="bar__fill" style="width:{(v / vmax) * 100:.1f}%"></div></div>'
                f'<div class="bar__val mono">{_num(v, 5)}</div></div>' for f, v in top)
            body += f'<div class="bars">{bars}</div>'
        loc = shap.get("local_example")
        if loc and loc.get("top_contributions"):
            contribs = "".join(
                f"<li><span class='mono'>{esc(f)}</span> "
                f"<span class='mono {'pos' if v >= 0 else 'neg'}'>{'+' if v >= 0 else ''}{_num(v, 4)}</span></li>"
                for f, v in loc["top_contributions"][:6])
            body += _callout(
                f"One worked example (row #{esc(loc.get('index'))}, predicted "
                f"{esc(loc.get('predicted'))})",
                "<p>The features that pushed this single prediction the most (＋ toward the "
                f"positive class, − away):</p><ul class='contribs'>{contribs}</ul>", "note")

    rel = e.get("reliability")
    if rel:
        body += "<h3>When to trust it: confidence-based triage</h3>"
        body += ("<p class='muted'>If we route the least-confident predictions to a human and "
                 "auto-accept the rest, how many mistakes does that catch?</p>")
        stat_rows = [
            ["Overall accuracy", _num(rel.get("overall_accuracy"))],
            [f"Flag the least-confident", _pct(rel.get("coverage_flagged"), 0)],
            ["→ errors caught by flagging",
             f"{_pct(rel.get('error_capture_rate'), 0)} "
             f"({_num(rel.get('errors_caught_by_flag'))} of {_num(rel.get('errors_total'))})"],
            ["Accuracy of auto-accepted slice", _num(rel.get("accuracy_auto_accepted"))],
            ["Accuracy of flagged slice", _num(rel.get("accuracy_flagged"))],
        ]
        body += '<div class="grid2">' + _table(["Reliability", "Value"], stat_rows, right_from=1)
        fig = _embed_figure(output_dir, "reliability", "Reliability curve: confidence vs accuracy")
        body += (fig or "") + "</div>"
        if e.get("reliability_note"):
            body += _callout("In plain words", f"<p>{esc(e['reliability_note'])}.</p>", "good")
    return _section("s4", "S4 · Explain", "Explainability & reliability", body)


def _sec_drift(ctx: dict, output_dir: str, task: str) -> str:
    d = ctx.get("drift", {})
    if d.get("status") != "ok":
        note = _skipped_note(d, "Drift")
        sig = (d or {}).get("significance")
        extra = _significance_block(sig, task) if sig else ""
        return _section("s5", "S5 · Drift", "Concept-drift monitoring & significance",
                        note + extra)
    metric = d.get("metric", "score")
    intro = (
        f"<p>A model is trained once on the past — does it keep working as time moves on? We "
        f"replay the test period in <b>{_num(d.get('n_chunks'))} time slices</b> and track a "
        f"<b>frozen</b> model against an <b>adaptive</b> one that retrains as data arrives "
        f"(<span class='mono'>{esc(d.get('retrain_mode'))}</span> window). We also measure "
        f"<b>input drift</b> directly with PSI, independent of accuracy.</p>"
    )
    body = intro
    body += _embed_figure(output_dir, "drift_performance",
                          f"{metric.title()} over time: frozen vs adaptive") or ""
    body += _embed_figure(output_dir, "drift_psi",
                          "Feature drift (mean PSI) over time vs the training window") or ""

    trend = d.get("trend", {})
    fd = d.get("feature_drift", {})
    facts = [
        [f"Frozen {metric} (early → late)",
         f"{_num(trend.get('early_mean'))} → {_num(trend.get('late_mean'))}"],
        ["Drift magnitude (>0 = worsened)", _num(trend.get("drift_magnitude"))],
        [f"Frozen mean {metric}", _num(d.get("static_mean"))],
        [f"Adaptive mean {metric}", _num(d.get("adaptive_mean"))],
        ["Peak input drift (max mean PSI)", _num(fd.get("max_mean_psi"))],
    ]
    det = d.get("detectors", {})
    if det.get("status") == "ok":
        facts.append(["ADWIN change-points detected", _num(det.get("n_changes"))])
    body += '<div class="grid2">' + _table(["Drift metric", "Value"], facts, right_from=1)
    # per-chunk table (compact)
    chunks = d.get("chunks", [])
    if chunks:
        headers = ["Chunk", f"Frozen {metric}", f"Adaptive {metric}", "Mean PSI", "Top drifting feature"]
        rows = []
        for c in chunks:
            tdf = c.get("top_drift_feature") or ["—", None]
            rows.append([
                _num(c.get("chunk")), _num(c.get("static_score")),
                _num(c.get("adaptive_score")), _num(c.get("mean_psi")),
                f"{esc(tdf[0])} ({_num(tdf[1])})",
            ])
        body += _table(headers, rows, right_from=1)
    body += "</div>"
    if d.get("verdict"):
        body += _callout("Verdict", f"<p>{esc(d['verdict'])}</p>", "note")

    body += _significance_block(d.get("significance"), task)
    return _section("s5", "S5 · Drift", "Concept-drift monitoring & significance", body)


def _significance_block(sig: Optional[dict], task: str) -> str:
    if not sig:
        return ""
    test_name = "McNemar's test" if task == "classification" else "Wilcoxon signed-rank test"
    intro = (
        f"<h3>Statistical significance ({test_name})</h3>"
        "<p class='muted'>A better score can be luck. These tests check whether each "
        "head-to-head difference is real (p &lt; 0.05) or within noise.</p>"
    )
    rows = []
    for label, r in sig.items():
        pretty = label.replace("_", " ")
        rows.append([
            esc(pretty),
            esc(r.get("better_model", "—")),
            _num(r.get("p_value"), 4),
            "significant" if r.get("significant") else "not significant",
        ])
    tbl = _table(["Comparison", "Better", "p-value", "Result"], rows, right_from=2)
    # surface the interpretations
    notes = "".join(f"<li>{esc(r.get('interpretation', ''))}</li>"
                    for r in sig.values() if r.get("interpretation"))
    body = intro + tbl
    if notes:
        body += f"<ul class='notes'>{notes}</ul>"
    return body


def _sec_howto(cfg: Optional[Any], ctx: dict) -> str:
    task = ctx.get("task", "your target")
    example = (
        "dataset:\n"
        "  source: csv          # or 'openml'\n"
        "  path: data/your.csv   # your file\n"
        "  target: label         # the column to predict\n"
        "  time_col: timestamp   # column giving time order (or null)\n"
        "split:  {train_frac: 0.6, test_frac: 0.4}\n"
        "models: {boosters: [xgboost, lightgbm, catboost, random_forest], stacking: true}\n"
        "tuning: {budget_trials: 40, cv_splits: 3, scoring: null}\n"
        "drift:  {n_chunks: 10, retrain: sliding}   # none | periodic | sliding\n"
        "report: {title: My Report, fbeta: 2.0}\n"
        "task: auto             # auto | classification | regression\n"
        "random_state: 42\n"
    )
    knobs = _table(
        ["Setting", "What it controls"],
        [
            ["dataset.source / path / name", "Point at a local CSV or an OpenML dataset."],
            ["dataset.target", "The column to predict. Task is auto-detected from it."],
            ["dataset.time_col", "Set it and the split becomes chronological and drift runs; "
                                 "leave it null for a random split (drift is skipped)."],
            ["split.train_frac / test_frac", "How much data trains vs. tests."],
            ["models.boosters", "Which ensembles to try; uninstalled ones skip gracefully."],
            ["tuning.budget_trials", "Search budget for Random & Bayesian (kept equal for fairness)."],
            ["tuning.cv_splits / scoring", "Cross-validation folds and the metric optimized."],
            ["drift.n_chunks / retrain", "Time slices, and whether/how the adaptive model retrains."],
            ["report.fbeta", "Weighting of recall vs precision in the Fβ score."],
        ],
        right_from=1)
    steps = (
        "<ol class='steps'>"
        "<li><b>Install</b> once: <span class='mono'>pip install -r requirements.txt</span>. "
        "Optional libraries (xgboost, lightgbm, catboost, optuna, shap, river, statsmodels) each "
        "degrade gracefully if absent.</li>"
        "<li><b>Edit</b> <span class='mono'>config.yaml</span> — point <span class='mono'>dataset</span> "
        "at your file and name your <span class='mono'>target</span> column. Usually nothing else needs to change.</li>"
        "<li><b>Run</b> <span class='mono'>python run.py --config config.yaml</span>.</li>"
        "<li><b>Read</b> the fresh <span class='mono'>results/report.html</span> — this same "
        "report, rebuilt for your data.</li></ol>"
    )
    interpreting = _callout(
        "How to read your results",
        "<p>Start at the verdict panel up top. Check whether a simple baseline is close to the "
        "best ensemble (if so, prefer the simple one). In tuning, weigh the score gain against the "
        "wall-time cost. For drift, remember <b>input drift (PSI) is not the same as performance "
        "drift</b> — PSI &gt; 0.2 means inputs moved, but only a falling accuracy/rising-RMSE "
        "curve means the model is actually decaying. Trust the significance tests over raw decimal "
        "gaps.</p>", "note")
    body = (
        f"<p>TabML-Guard is dataset-agnostic: everything above was produced from a config file and "
        f"your data — nothing here is hard-coded to this particular dataset (this run was "
        f"<b>{esc(task)}</b>). To run it on your own tabular data:</p>"
        + steps
        + "<h3>The config file</h3><pre class='code'>" + esc(example) + "</pre>"
        + knobs + interpreting
    )
    return _section("howto", "Guide", "Run this on your own dataset", body)


def _sec_glossary() -> str:
    items = [
        ("ROC-AUC / PR-AUC", "Ranking quality of a classifier from 0.5 (random) to 1.0 (perfect). "
                             "PR-AUC is more informative when classes are imbalanced."),
        ("Fβ score", "A blend of precision and recall; β &gt; 1 weights recall more heavily."),
        ("Regularization (L1/L2/ElasticNet)", "Penalties that keep a model simple. L1 zeroes out "
                                              "weak features; L2 shrinks all of them; ElasticNet blends both."),
        ("SHAP", "A method that attributes each prediction to its input features, with a sound "
                 "game-theory basis — used here for global importance and one worked example."),
        ("Reliability / error capture", "Using the model's own confidence to flag shaky predictions "
                                        "for review; error capture is the share of mistakes caught by that flag."),
        ("PSI (Population Stability Index)", "How far a feature's distribution has moved from training. "
                                            "&gt;0.1 moderate, &gt;0.2 large shift."),
        ("Prequential / walk-forward", "Test-then-train evaluation over time: score a slice, then let "
                                       "the adaptive model learn it — no peeking at the future."),
        ("McNemar's test", "Compares two classifiers on the rows where they disagree; a small p-value "
                           "means the winner's edge is real, not noise."),
    ]
    body = "<dl class='glossary'>" + "".join(
        f"<dt>{esc(t)}</dt><dd>{esc(d)}</dd>" for t, d in items) + "</dl>"
    return _section("glossary", "Reference", "Glossary", body)


def _config_block(cfg: Optional[Any], ctx: dict) -> str:
    """A compact 'this run's settings' readout, from cfg if available else context."""
    tuning = ctx.get("tuning") or {}
    drift = ctx.get("drift") or {}
    rows = []
    if cfg is not None:
        try:
            rows = [
                ["Source", f"{esc(cfg.dataset.source)} / {esc(cfg.dataset.name or cfg.dataset.path)}"],
                ["Split", f"train {cfg.split.train_frac:g} / test {cfg.split.test_frac:g}"],
                ["Boosters", esc(", ".join(cfg.models.boosters))],
                ["Tuning", f"{cfg.tuning.budget_trials} trials, {cfg.tuning.cv_splits}-fold"],
                ["Drift", f"{cfg.drift.n_chunks} chunks, retrain={esc(cfg.drift.retrain)}"],
                ["Random state", _num(cfg.random_state)],
            ]
        except Exception:
            rows = []
    if not rows:  # derive from context
        rows = [
            ["Tuning budget", f"{_num(tuning.get('budget_trials'))} trials, "
                              f"{_num(tuning.get('cv_splits'))}-fold"],
            ["Drift", f"{_num(drift.get('n_chunks'))} chunks, "
                      f"retrain={esc(drift.get('retrain_mode', '—'))}"],
        ]
    return "<h3>This run's configuration</h3>" + _table(["Setting", "Value"], rows, right_from=1)


# --------------------------------------------------------------------------- #
# CSS (inline, self-contained)
# --------------------------------------------------------------------------- #
_CSS = """
:root{
  --ink:#14213d; --paper:#ffffff; --panel:#f6f8fb; --panel2:#eef2f8;
  --accent:#0f766e; --accent-weak:#d6eae7; --muted:#5b6472; --line:#e3e8ef;
  --good:#15803d; --caution:#b45309; --alert:#b91c1c; --neutral:#64748b;
  --good-bg:#e8f5ec; --caution-bg:#fdf3e6; --alert-bg:#fbeaea; --neutral-bg:#eef1f5;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--panel);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.6;font-size:16px;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.wrap{max-width:940px;margin:0 auto;padding:0 24px}
a{color:var(--accent);text-decoration:none}
h1,h2,h3{line-height:1.25;letter-spacing:-.02em;margin:0}
h2{font-size:1.55rem;margin:0} h3{font-size:1.08rem;margin:1.6em 0 .5em}
p{margin:.6em 0} .muted{color:var(--muted);font-size:.94rem}

/* top bar */
.topbar{position:sticky;top:0;z-index:20;background:rgba(20,33,61,.97);color:#fff;
  border-bottom:2px solid var(--accent)}
.topbar .wrap{display:flex;align-items:center;gap:18px;height:52px}
.topbar__brand{font-weight:700;letter-spacing:-.02em}
.topbar__brand b{color:#7fd8cf}
.topbar__nav{display:flex;gap:14px;margin-left:auto;flex-wrap:wrap}
.topbar__nav a{color:#cfd8e6;font-size:.82rem;text-transform:uppercase;letter-spacing:.06em}
.topbar__nav a:hover{color:#fff}

/* hero / verdict readout */
.hero{background:linear-gradient(180deg,#14213d,#1b2b4d);color:#eaf0f9;padding:40px 0 34px}
.hero h1{font-size:2.1rem;color:#fff;margin-bottom:6px}
.hero .sub{color:#a9b6cc;font-size:1rem}
.readout{margin-top:26px;background:rgba(255,255,255,.05);border:1px solid rgba(127,216,207,.25);
  border-radius:14px;padding:22px 24px;display:grid;grid-template-columns:auto 1fr;gap:26px}
.readout__score{border-right:1px solid rgba(255,255,255,.14);padding-right:26px;min-width:170px}
.readout__label{font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;color:#7fd8cf}
.readout__model{font-size:1.15rem;font-weight:700;color:#fff;margin:4px 0 10px;word-break:break-word}
.readout__big{font-size:2.6rem;font-weight:700;color:#fff;line-height:1}
.readout__metric{font-size:.8rem;color:#a9b6cc;margin-top:2px}
.pills{display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start}
.pill{font-size:.78rem;font-weight:600;padding:5px 11px;border-radius:999px;
  border:1px solid transparent;white-space:nowrap}
.pill--accent{background:var(--accent-weak);color:#0b544d;border-color:#a9d6cf}
.pill--good{background:var(--good-bg);color:var(--good);border-color:#b6ddc2}
.pill--caution{background:var(--caution-bg);color:var(--caution);border-color:#eecfa6}
.pill--alert{background:var(--alert-bg);color:var(--alert);border-color:#eebcbc}
.pill--neutral{background:var(--neutral-bg);color:var(--neutral);border-color:#d3dae4}
.rec{margin-top:20px;background:#fff;color:var(--ink);border-radius:12px;padding:18px 20px;
  border-left:4px solid var(--accent)}
.rec .rec__label{font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);
  font-weight:700;margin-bottom:4px}

/* sections */
.section{background:var(--paper);border:1px solid var(--line);border-radius:14px;
  padding:26px 28px;margin:22px 0}
.section__head{margin-bottom:14px}
.eyebrow{display:inline-block;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.74rem;
  font-weight:700;letter-spacing:.1em;color:var(--accent);background:var(--accent-weak);
  padding:3px 9px;border-radius:6px;margin-bottom:8px}
.section__sub{color:var(--muted);margin-top:2px}
.content-first{margin-top:22px}

/* tables */
.tablewrap{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line)}
th{background:var(--panel2);font-size:.76rem;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);font-weight:700;white-space:nowrap}
td.num,th.num{text-align:right}
tbody tr:last-child td{border-bottom:none}
tbody tr:nth-child(even){background:#fbfcfe}
tr.row--best{background:var(--accent-weak)!important}
tr.row--best td:first-child::after{content:" \\2605";color:var(--accent)}
.pos{color:var(--good)} .neg{color:var(--alert)}

/* callouts */
.callout{border-radius:10px;padding:14px 16px;margin:14px 0;border:1px solid var(--line);
  background:var(--panel)}
.callout--note{border-left:4px solid var(--accent)}
.callout--good{border-left:4px solid var(--good);background:var(--good-bg)}
.callout--neutral{border-left:4px solid var(--neutral)}
.callout__title{font-weight:700;font-size:.9rem;margin-bottom:3px}
.callout__body{font-size:.93rem} .callout__body p{margin:.3em 0}

/* bars */
.bars{display:flex;flex-direction:column;gap:7px;margin:12px 0}
.bar{display:grid;grid-template-columns:130px 1fr 120px;align-items:center;gap:10px;font-size:.86rem}
.bar__label{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.8rem;color:var(--ink);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar__track{background:var(--panel2);border-radius:5px;height:14px;overflow:hidden}
.bar__fill{background:var(--accent);height:100%;border-radius:5px}
.bar__val{text-align:right;color:var(--muted);font-size:.8rem}

/* figures + svg */
.fig{margin:14px 0;text-align:center}
.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}
.fig figcaption{font-size:.8rem;color:var(--muted);margin-top:6px}
.svgchart{width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:10px;margin:12px 0}
.svg-title{font-size:11px;fill:var(--ink);font-weight:600}
.svg-grid{stroke:var(--line);stroke-width:1}
.svg-ytick{font-size:9px;fill:var(--muted);text-anchor:end;font-family:ui-monospace,monospace}
.svg-xtick{font-size:9px;fill:var(--muted);font-family:ui-monospace,monospace}
.svg-xtick--end{text-anchor:end}
.svg-legend{font-size:10px;fill:var(--ink)}
.svg-ylabel{font-size:9px;fill:var(--muted);text-anchor:middle}

/* misc content */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}
.code{background:#14213d;color:#e6edf7;border-radius:10px;padding:16px 18px;overflow-x:auto;
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.82rem;line-height:1.5;white-space:pre}
.steps,.notes,.contribs{padding-left:20px} .steps li,.notes li{margin:.4em 0}
.contribs{list-style:none;padding:0} .contribs li{display:flex;justify-content:space-between;
  border-bottom:1px dashed var(--line);padding:4px 0;max-width:340px}
.glossary{margin:0} .glossary dt{font-weight:700;margin-top:12px}
.glossary dd{margin:2px 0 0;color:var(--muted);font-size:.92rem}
footer{color:var(--muted);font-size:.82rem;text-align:center;padding:30px 0 44px}

@media (max-width:720px){
  .readout{grid-template-columns:1fr} .readout__score{border-right:none;
    border-bottom:1px solid rgba(255,255,255,.14);padding:0 0 16px}
  .grid2{grid-template-columns:1fr} .bar{grid-template-columns:100px 1fr 90px}
}
@media print{.topbar{position:static} body{background:#fff} .section{break-inside:avoid}}
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _dataset_slug(cfg: Optional[Any], context: dict, title: str) -> str:
    """A filesystem-safe dataset identifier for the report filename.

    Prefers the configured OpenML name, else the CSV file stem, else the OpenML
    data_id, else the target column, else the title. Guarantees a non-empty slug.
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
        raw = (context.get("profile", {}) or {}).get("target") or title or "report"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(raw)).strip("-").lower()
    return slug or "dataset"


def build(context: dict, output_dir: str, title: str = "TabML-Guard Report",
          cfg: Optional[Any] = None) -> str:
    """Assemble a self-contained report.html from the run context. Returns its path."""
    task = context.get("task", "classification")
    primary = (context.get("baselines") or {}).get("primary_metric") \
        or ("roc_auc" if task == "classification" else "rmse")
    ins = _insights(context, task, primary)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    prof = context.get("profile", {})
    ds_name = ""
    if cfg is not None:
        try:
            ds_name = str(cfg.dataset.name or cfg.dataset.path or "")
        except Exception:
            ds_name = ""
    subtitle = (f"{esc(ds_name)} · " if ds_name else "") + \
        f"{_num(prof.get('n_rows'))} rows × {_num(prof.get('n_features'))} features · {esc(task)}"

    # verdict readout
    pills_html = "".join(_pill(t, k) for t, k in ins["pills"])
    rec_html = ""
    if ins["recommendation"]:
        rec_html = (f'<div class="rec"><div class="rec__label">Recommendation</div>'
                    f'<div>{ins["recommendation"]}</div></div>')
    complexity = ""
    if ins["complexity_note"]:
        complexity = _callout("Headline finding", f"<p>{esc(ins['complexity_note'])}</p>", "note")

    hero = f"""
    <header class="hero"><div class="wrap">
      <h1>{esc(title)}</h1>
      <div class="sub">{subtitle} &nbsp;·&nbsp; generated {esc(now)}</div>
      <div class="readout">
        <div class="readout__score">
          <div class="readout__label">Recommended model</div>
          <div class="readout__model mono">{esc(ins['champion'] or '—')}</div>
          <div class="readout__big mono">{_num(ins['champion_score'])}</div>
          <div class="readout__metric">{esc(_METRIC_LABEL.get(primary, primary))} on held-out test</div>
        </div>
        <div class="pills">{pills_html}</div>
      </div>
      {rec_html}
    </div></header>"""

    nav = """
    <div class="topbar"><div class="wrap">
      <div class="topbar__brand">TabML<b>·</b>Guard</div>
      <nav class="topbar__nav">
        <a href="#s0">Data</a><a href="#s1">Baselines</a><a href="#s2">Ensembles</a>
        <a href="#s3">Tuning</a><a href="#s4">Explain</a><a href="#s5">Drift</a>
        <a href="#howto">Use it</a>
      </nav>
    </div></div>"""

    sections = [
        f'<div class="content-first">{complexity}</div>' if complexity else "",
        _sec_dataset(context, _config_block(cfg, context)),
        _sec_baselines(context, task, primary),
        _sec_boosting(context, task, primary),
        _sec_tuning(context, task, primary),
        _sec_explain(context, output_dir),
        _sec_drift(context, output_dir, task),
        _sec_howto(cfg, context),
        _sec_glossary(),
    ]
    body = nav + hero + '<main class="wrap">' + "".join(sections) + "</main>"
    footer = ('<footer>Generated by TabML-Guard — a configurable pipeline for explainable, '
              'drift-aware tabular prediction. This report is self-contained; every number is '
              'reproducible from <span class="mono">results/tables/results.json</span>.</footer>')

    doc = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>{esc(title)}</title><style>{_CSS}</style></head>"
           f"<body>{body}{footer}</body></html>")

    slug = _dataset_slug(cfg, context, title)
    out = Path(output_dir) / f"report_{slug}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return str(out)
