"""Stage 5 — Concept-drift monitoring + statistical significance (the differentiator).

The plain-English question: **a model is trained once on the past — does it keep
working as time moves on, or does the world shift under it?** And if we retrain as
new data arrives, is the retrained model *significantly* better than the frozen one?

What this stage does, step by step:

  1. **Walk forward through time.** The held-out test set is time-ordered, so we cut
     it into ``n_chunks`` consecutive slices ("this week, next week, ..."). We score
     the frozen (train-once) model on each slice in turn — a *prequential* /
     test-then-train evaluation. A downward trend = the model is ageing.

  2. **Retrain and race.** In parallel we keep an *adaptive* copy that re-fits as each
     slice goes by — either an **expanding** window (``periodic``: keep all history)
     or a fixed **sliding** window (``sliding``: only recent history). Predicting each
     slice happens *before* that slice is learned from, so there is no look-ahead.

  3. **Measure input drift directly.** For every numeric feature we compute the
     Population Stability Index (PSI) of each slice against the training window. PSI
     tells us the *inputs* moved, independent of labels (>0.1 = moderate, >0.2 = large).

  4. **Detect change points.** If the ``river`` library is present, ADWIN watches the
     frozen model's error stream and flags where its accuracy statistically shifts.
     Absent river, the per-chunk accuracy curve already tells the story.

  5. **Test significance.** McNemar's test (classification) / Wilcoxon (regression),
     via ``tabml_guard.stats``, on (a) champion vs the runner-up model, (b) champion vs the
     simple linear baseline, and (c) frozen vs adaptive — so every "X beats Y" claim
     in the report is backed by a p-value, not just a decimal.

Draws on Paper 4 (Concept Drift with Continuous Kernel Learning, Information Sciences
2024, DOI 10.1016/j.ins.2024.120649): the streaming/distribution-shift framing and the
Electricity dataset. We implement the practical monitor-and-retrain pattern rather than
their continuous-kernel algorithm.

Robustness choices worth knowing
--------------------------------
* **Skips cleanly** when there is no time column (drift is meaningless on a random split).
* **No look-ahead.** The adaptive model predicts a chunk strictly before training on it.
* **Leakage-safe retrain.** Each refit clones the *whole* pipeline (``sklearn.clone``),
  so the preprocessor's imputation/scaling statistics are relearned on the new window.
* **Never raises.** Retraining, detectors, PSI and plotting are each wrapped; a failure
  degrades to a partial-but-valid summary instead of sinking the run.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config
from .data import DataBundle
from . import stats


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _score(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> float:
    """Per-chunk headline score: accuracy (classification) or RMSE (regression)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if task == "classification":
        return float(np.mean(y_pred == y_true))
    return float(np.sqrt(np.mean((y_pred.astype(float) - y_true.astype(float)) ** 2)))


def _higher_is_better(task: str) -> bool:
    return task == "classification"          # accuracy up good; rmse down good


def _positive_scores(y_proba) -> Optional[np.ndarray]:
    """Positive-class score from a (n,2) matrix or a 1-D score, for per-chunk AUC."""
    if y_proba is None:
        return None
    p = np.asarray(y_proba, dtype=float)
    if p.ndim == 2:
        return p[:, 1] if p.shape[1] >= 2 else p.ravel()
    return p.ravel()


def _chunk_auc(scores: Optional[np.ndarray], y_true: np.ndarray) -> Optional[float]:
    """ROC-AUC on one chunk, or None if undefined (single class / no scores)."""
    if scores is None:
        return None
    yt = np.asarray(y_true).ravel()
    if len(np.unique(yt)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return round(float(roc_auc_score(yt, scores)), 4)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Feature drift — Population Stability Index
# --------------------------------------------------------------------------- #
def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI of ``actual`` vs a reference ``expected`` distribution (one feature).

    Bins are quantile edges of the reference; both distributions are turned into
    proportions and compared as sum((a-e)*ln(a/e)). Near-constant features -> 0.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 1:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                        # (near-)constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    eps = 1e-6
    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), eps, None)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def _chunk_feature_drift(bundle: DataBundle, chunk_idx: np.ndarray) -> tuple[float, Optional[list]]:
    """(mean PSI across numeric features, worst [feature, psi]) for one test chunk."""
    num_cols = bundle.numeric_cols
    if not num_cols:
        return 0.0, None
    per_feat = []
    for c in num_cols:
        ref = bundle.X_train[c].to_numpy()
        cur = bundle.X_test[c].iloc[chunk_idx].to_numpy()
        per_feat.append((c, _psi(ref, cur)))
    mean_psi = float(np.mean([p for _, p in per_feat]))
    worst = max(per_feat, key=lambda kv: kv[1])
    return round(mean_psi, 4), [worst[0], round(float(worst[1]), 4)]


# --------------------------------------------------------------------------- #
# Adaptive retraining (expanding "periodic" or fixed "sliding" window)
# --------------------------------------------------------------------------- #
def _retrain_window(bundle: DataBundle, seen_end: int, mode: str):
    """Assemble the (X, y) training window that has been *seen* up to ``seen_end``.

    ``seen_end`` is an exclusive index into the test stream. The window always
    starts from the original training block and appends test rows already observed.
    ``sliding`` then keeps only the most recent ``len(X_train)`` rows.
    """
    import pandas as pd

    if seen_end <= 0:
        X = bundle.X_train.copy()
        y = np.asarray(bundle.y_train)
    else:
        X = pd.concat([bundle.X_train, bundle.X_test.iloc[:seen_end]], ignore_index=True)
        y = np.concatenate([np.asarray(bundle.y_train), np.asarray(bundle.y_test)[:seen_end]])
    if mode == "sliding":
        w = len(bundle.X_train)
        if len(X) > w:
            X = X.iloc[-w:].reset_index(drop=True)
            y = y[-w:]
    return X, y


def _prequential_adaptive(best_estimator, bundle: DataBundle, chunks: list, mode: str) -> Optional[dict]:
    """Walk chunks: predict-then-retrain. Returns per-chunk scores + full-stream preds.

    Predicts chunk *i* with a model trained only on data before chunk *i*, then
    refits (for the next chunk) on the window implied by ``mode``. Returns None if
    ``mode == 'none'`` or if the very first fit fails.
    """
    if mode == "none":
        return None
    try:
        from sklearn.base import clone
    except Exception:
        return None

    task = bundle.task
    scores, full_pred = [], np.empty(len(bundle.X_test), dtype=float)
    model = None
    try:
        for i, idx in enumerate(chunks):
            # Fit on everything seen strictly before this chunk.
            if model is None:
                Xw, yw = _retrain_window(bundle, int(idx[0]), mode)
                model = clone(best_estimator).fit(Xw, yw)

            Xc = bundle.X_test.iloc[idx]
            pred = np.asarray(model.predict(Xc)).ravel()
            if task == "classification":
                pred = pred.astype(int)
            full_pred[idx] = pred
            scores.append(round(_score(bundle.y_test[idx], pred, task), 4))

            # Retrain incorporating this chunk, ready for the next one (skip after last).
            if i < len(chunks) - 1:
                Xw, yw = _retrain_window(bundle, int(idx[-1]) + 1, mode)
                model = clone(best_estimator).fit(Xw, yw)
    except Exception as exc:
        print(f"[stage5] adaptive retraining skipped: {type(exc).__name__}: {exc}")
        return None

    fp = full_pred.astype(int) if task == "classification" else full_pred
    return {"scores": scores, "full_pred": fp, "mean": round(float(np.mean(scores)), 4)}


# --------------------------------------------------------------------------- #
# Change-point detection (optional: river ADWIN on the frozen error stream)
# --------------------------------------------------------------------------- #
def _detect_changes(correct_static: np.ndarray, chunks: list) -> dict:
    """ADWIN over the frozen model's 0/1 error stream; map change points to chunks."""
    try:
        from river.drift import ADWIN
    except Exception:
        return {"method": "adwin", "status": "unavailable (river not installed)"}
    try:
        err = (1 - np.asarray(correct_static).ravel().astype(int))
        det = ADWIN()
        positions = []
        for i, e in enumerate(err):
            det.update(int(e))
            if getattr(det, "drift_detected", False):
                positions.append(i)
        bounds = np.array([idx[-1] for idx in chunks])
        change_chunks = sorted({int(np.searchsorted(bounds, p)) for p in positions})
        return {
            "method": "adwin",
            "status": "ok",
            "n_changes": len(positions),
            "change_chunks": change_chunks,
        }
    except Exception as exc:
        return {"method": "adwin", "status": f"error: {type(exc).__name__}"}


# --------------------------------------------------------------------------- #
# Significance tests (via tabml_guard.stats)
# --------------------------------------------------------------------------- #
def _compare(y_true, a, b, name_a, name_b, task) -> dict:
    if task == "classification":
        return stats.mcnemar_compare(y_true, a, b, name_a, name_b)
    return stats.paired_error_test(y_true, a, b, name_a, name_b)


def _significance(ctx: dict, bundle: DataBundle, adaptive_full: Optional[np.ndarray]) -> dict:
    """McNemar/Wilcoxon on the report's key claims: champion vs others, static vs adaptive."""
    out: dict = {}
    task = bundle.task
    y = bundle.y_test
    champ = ctx.get("best_overall")
    if champ is None:
        return out

    results = ctx.get("results", {}) or {}
    others = [r for n, r in results.items() if n != champ.name]

    # Champion vs the next-best model overall ("is the winner's edge real?").
    if others:
        from .modeling import pick_best

        runner = pick_best(others, task)
        if runner is not None:
            out["champion_vs_runner_up"] = _compare(
                y, champ.y_pred, runner.y_pred, champ.name, runner.name, task)

    # Champion vs the simple regularized linear baseline ("is complexity worth it?").
    base = ctx.get("baseline_best")
    if base is not None and base.name != champ.name:
        out["champion_vs_linear_baseline"] = _compare(
            y, champ.y_pred, base.y_pred, champ.name, base.name, task)

    # Frozen vs adaptive ("did retraining significantly help?").
    if adaptive_full is not None and champ.y_pred is not None:
        out["static_vs_adaptive"] = _compare(
            y, champ.y_pred, adaptive_full, "static", "adaptive", task)
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _make_plots(chunk_ids, static_scores, adaptive_scores, psi_traj, metric_label, cfg) -> dict:
    figs: dict = {}
    try:
        from . import paths, plots

        run_out = str(paths.run_dir(cfg))
        series = {"frozen (train once)": static_scores}
        if adaptive_scores:
            series["adaptive (retrained)"] = adaptive_scores
        figs["drift_performance"] = plots.plot_lines(
            chunk_ids, series, run_out, "drift_performance",
            title="Performance over time (walk-forward)",
            xlabel="time chunk (old -> new)", ylabel=metric_label,
        )
        figs["drift_psi"] = plots.plot_lines(
            chunk_ids, {"mean PSI (input drift)": psi_traj}, run_out, "drift_psi",
            title="Feature drift over time (PSI vs training window)",
            xlabel="time chunk (old -> new)", ylabel="mean PSI",
        )
    except Exception as exc:
        print(f"[stage5] plotting skipped: {type(exc).__name__}: {exc}")
    return figs


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def _verdict(task, static_scores, adaptive_mean, static_mean, drift_mag,
             worse_over_time, max_psi, sig: dict, mode: str) -> str:
    metric = "accuracy" if task == "classification" else "RMSE"
    first, last = static_scores[0], static_scores[-1]
    trend = "declined" if worse_over_time else "held up"
    parts = [
        f"Frozen-model {metric} {trend} across time ({first:.3f} -> {last:.3f}; "
        f"drift magnitude {drift_mag:+.3f}). Peak input drift (mean PSI) = {max_psi:.3f} "
        f"({'notable' if max_psi > 0.2 else 'mild' if max_psi > 0.1 else 'low'})."
    ]
    if adaptive_mean is not None:
        better = adaptive_mean > static_mean if task == "classification" else adaptive_mean < static_mean
        gap = abs(adaptive_mean - static_mean)
        sva = sig.get("static_vs_adaptive", {})
        sigtxt = ""
        if sva:
            sigtxt = (" and the difference is statistically significant"
                      if sva.get("significant") else " but not statistically significant")
        parts.append(
            f"A {mode}-retrained model {'beat' if better else 'did not beat'} the frozen one "
            f"({metric} {adaptive_mean:.3f} vs {static_mean:.3f}, gap {gap:.3f}{sigtxt})."
        )
    cvb = sig.get("champion_vs_linear_baseline")
    if cvb:
        parts.append(
            f"Champion vs linear baseline: {'significant' if cvb.get('significant') else 'not significant'} "
            f"(p={cvb.get('p_value')})."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(bundle: DataBundle, cfg: Config, ctx: Optional[dict] = None) -> dict:
    """Temporal drift analysis + significance tests. Never raises; degrades gracefully."""
    ctx = ctx if ctx is not None else {}
    if not bundle.has_time:
        # Drift needs chronological order; a random split can't simulate deployment.
        # We can still report significance between models on the test set.
        sig = _significance(ctx, bundle, adaptive_full=None)
        return {"status": "skipped", "stage": "drift", "reason": "no time column",
                "significance": sig}

    best = ctx.get("best_overall")
    if best is None:
        print("[stage5] no best model in context — skipped.")
        return {"status": "skipped", "stage": "drift", "reason": "no best model"}

    task = bundle.task
    mode = cfg.drift.retrain
    n_chunks = int(cfg.drift.n_chunks)
    n_test = len(bundle.X_test)
    if n_test < n_chunks:
        n_chunks = max(2, n_test)
    chunks = [c for c in np.array_split(np.arange(n_test), n_chunks) if len(c)]
    print(f"[stage5] walking '{best.name}' across {len(chunks)} time chunks (retrain='{mode}') ...")

    # --- Frozen model: reuse the predictions/probabilities Stage 2/3 already made. ---
    static_pred_full = np.asarray(best.y_pred).ravel()
    if task == "classification":
        static_pred_full = static_pred_full.astype(int)
    pos_scores = _positive_scores(best.y_proba)
    y_test = np.asarray(bundle.y_test).ravel()
    correct_static = (static_pred_full == y_test).astype(int) if task == "classification" else None

    # --- Adaptive model (predict-then-retrain), if requested. ---
    adaptive = _prequential_adaptive(best.estimator, bundle, chunks, mode)

    # --- Per-chunk table: scores, feature drift, timestamps. ---
    time_test = bundle.time_test
    chunk_rows, static_scores, psi_traj = [], [], []
    for i, idx in enumerate(chunks):
        s_score = round(_score(y_test[idx], static_pred_full[idx], task), 4)
        static_scores.append(s_score)
        mean_psi, worst = _chunk_feature_drift(bundle, idx)
        psi_traj.append(mean_psi)
        row = {
            "chunk": i,
            "n": int(len(idx)),
            "static_score": s_score,
            "mean_psi": mean_psi,
            "top_drift_feature": worst,
        }
        if adaptive:
            row["adaptive_score"] = adaptive["scores"][i]
        if task == "classification":
            row["static_auc"] = _chunk_auc(pos_scores[idx] if pos_scores is not None else None, y_test[idx])
        if time_test is not None:
            row["time_start"] = str(time_test.iloc[int(idx[0])])
            row["time_end"] = str(time_test.iloc[int(idx[-1])])
        chunk_rows.append(row)

    # --- Trend / degradation (robust: first-third vs last-third means). ---
    third = max(1, len(static_scores) // 3)
    early = float(np.mean(static_scores[:third]))
    late = float(np.mean(static_scores[-third:]))
    hib = _higher_is_better(task)
    # Positive drift_mag always means "got worse over time", regardless of metric.
    drift_mag = round((early - late) if hib else (late - early), 4)
    worse_over_time = drift_mag > 0
    static_mean = round(float(np.mean(static_scores)), 4)
    max_psi = round(float(np.max(psi_traj)) if psi_traj else 0.0, 4)

    # --- Detectors + significance + plots. ---
    detectors = _detect_changes(correct_static, chunks) if correct_static is not None else \
        {"method": "adwin", "status": "n/a (regression)"}
    adaptive_full = adaptive["full_pred"] if adaptive else None
    sig = _significance(ctx, bundle, adaptive_full)
    metric_label = "accuracy" if task == "classification" else "RMSE"
    figs = _make_plots([r["chunk"] for r in chunk_rows], static_scores,
                       adaptive["scores"] if adaptive else None, psi_traj, metric_label, cfg)

    verdict = _verdict(task, static_scores, adaptive["mean"] if adaptive else None,
                       static_mean, drift_mag, worse_over_time, max_psi, sig, mode)
    print(f"[stage5] {verdict}")

    return {
        "status": "ok",
        "stage": "drift",
        "model": best.name,
        "metric": metric_label,
        "retrain_mode": mode,
        "n_chunks": len(chunks),
        "chunks": chunk_rows,
        "static_mean": static_mean,
        "adaptive_mean": adaptive["mean"] if adaptive else None,
        "trend": {
            "early_mean": round(early, 4),
            "late_mean": round(late, 4),
            "drift_magnitude": drift_mag,          # >0 => performance worsened over time
            "worse_over_time": bool(worse_over_time),
        },
        "feature_drift": {
            "mean_psi_trajectory": psi_traj,
            "max_mean_psi": max_psi,
        },
        "detectors": detectors,
        "significance": sig,
        "verdict": verdict,
        "figures": figs,
    }
