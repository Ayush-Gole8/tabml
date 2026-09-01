"""Stage 4 — SHAP explainability + a reliability layer.

Two questions this stage answers for a non-expert reader:

  1. **"Why does the model decide what it decides?"**  Global feature importance
     (which inputs matter most, overall) plus one worked local example (how a
     single prediction was built up from its features), computed with SHAP.

  2. **"When should we trust a prediction — and when should a human look?"**  A
     reliability layer that uses the model's own confidence to flag shaky
     predictions, then *proves* the flag works: the flagged set really does carry
     most of the errors. This is the practical, deployable half of Paper 3.

Draws on Paper 3 (SHAP Error Compensation, IEEE TAI 2025, DOI
10.1109/TAI.2024.3455313): SHAP is not just for pretty plots — the signal can be
used to identify unreliable predictions. We implement the *identify* half and
express it as a "send the unsure ones to human review" workflow.

Robustness choices worth knowing
---------------------------------
* **Model-type aware.** SHAP's fast ``TreeExplainer`` only works on tree models.
  The best model across all stages might be a *linear* baseline (it was, in the
  first ELEC2 run!), so we route linear models to ``LinearExplainer`` and anything
  else to the model-agnostic explainer, falling back gracefully if SHAP is absent.
* **Explain in the transformed space.** The model lives behind a preprocessor, so
  SHAP is computed on the *transformed* features and labelled with the
  preprocessor's output names, so importances line up with real column names.
* **Sampled for speed.** Global SHAP uses a bounded sample of the test set; a
  laptop stays responsive and the estimates are stable.
* **Reliability needs no SHAP.** The confidence-based reliability analysis uses
  probabilities we already computed, so it still runs even if SHAP is unavailable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import Config
from src.data import DataBundle
from src.modeling import ModelResult, feature_names_out


# --------------------------------------------------------------------------- #
# SHAP helpers
# --------------------------------------------------------------------------- #
def _shap_matrix(shap_values, positive_class: int = 1) -> np.ndarray:
    """Normalize SHAP output to a plain (n_samples, n_features) matrix.

    SHAP's shape depends on version + model: it can be a list (one array per
    class), an ``Explanation`` object, a 2-D array, or a 3-D
    (n, features, classes) array. We collapse all of these to one 2-D matrix for
    the class we care about (the positive class for binary problems).
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[positive_class] if len(shap_values) > positive_class else shap_values[-1]
    vals = getattr(shap_values, "values", shap_values)
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 3:  # (n, features, classes)
        idx = positive_class if vals.shape[-1] > positive_class else vals.shape[-1] - 1
        vals = vals[..., idx]
    return vals


def _build_explainer(model, X_bg: np.ndarray):
    """Pick the right SHAP explainer for the model family. Returns (explainer, kind)."""
    import shap

    if hasattr(model, "feature_importances_"):          # tree ensembles
        return shap.TreeExplainer(model), "tree"
    if hasattr(model, "coef_"):                          # linear models
        return shap.LinearExplainer(model, X_bg), "linear"
    # model-agnostic fallback (slower; use a tiny background)
    predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    return shap.Explainer(predict, X_bg[: min(100, len(X_bg))]), "kernel"


def _shap_analysis(best: ModelResult, bundle: DataBundle, cfg: Config,
                   max_samples: int = 800) -> Optional[dict]:
    """Global importance (mean |SHAP|) + one local example, best-effort."""
    try:
        import shap  # noqa: F401

        pipe = best.estimator
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
        names = feature_names_out(pipe)

        # Explain in transformed space, on a bounded sample of the test set.
        n = min(max_samples, len(bundle.X_test))
        X_sample = bundle.X_test.iloc[:n]
        X_trans = np.asarray(pre.transform(X_sample), dtype=float)
        bg = X_trans[: min(200, len(X_trans))]

        explainer, kind = _build_explainer(model, bg)
        raw = explainer.shap_values(X_trans) if hasattr(explainer, "shap_values") else explainer(X_trans)
        mat = _shap_matrix(raw)                       # (n, n_features)

        if mat.shape[1] != len(names):
            names = [f"f{i}" for i in range(mat.shape[1])]

        mean_abs = np.abs(mat).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        top = [[names[i], round(float(mean_abs[i]), 6)] for i in order]

        # One local example: the instance whose prediction leaned hardest.
        row_strength = np.abs(mat).sum(axis=1)
        j = int(np.argmax(row_strength))
        contribs = sorted(
            ([names[k], round(float(mat[j, k]), 6)] for k in range(mat.shape[1])),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:8]

        return {
            "method": kind,
            "n_explained": int(n),
            "global_importance": top,      # [[feature, mean|shap|], ...] best-first
            "local_example": {
                "index": j,
                "predicted": int(best.y_pred[j]) if j < len(best.y_pred) else None,
                "top_contributions": contribs,   # signed: + pushes toward positive class
            },
        }
    except Exception as exc:
        print(f"[stage4] SHAP skipped: {type(exc).__name__}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Reliability layer — confidence-based "flag the unsure ones" (no SHAP required)
# --------------------------------------------------------------------------- #
def _confidence(y_proba: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Per-instance confidence in [0,1]: max class probability (binary or multiclass)."""
    if y_proba is None:
        return None
    p = np.asarray(y_proba, dtype=float)
    if p.ndim == 2:
        return p.max(axis=1)
    if p.ndim == 1 and np.nanmin(p) >= 0.0 and np.nanmax(p) <= 1.0:
        return np.maximum(p, 1.0 - p)     # 1-D positive-class probability
    return None                            # decision scores — not a probability


def _reliability(best: ModelResult, bundle: DataBundle, n_bins: int = 10,
                 flag_frac: float = 0.20) -> Optional[dict]:
    """Does low confidence predict wrongness? Bin accuracy by confidence + a flag rule.

    The headline, deployable result: if we auto-accept the most-confident
    predictions and route the least-confident ``flag_frac`` to human review, how
    much of the total error do we catch, and how much cleaner is the auto-accepted
    slice?
    """
    if bundle.task != "classification":
        return None
    conf = _confidence(best.y_proba)
    if conf is None:
        print("[stage4] reliability skipped: model exposes no probabilities.")
        return None

    y_true = np.asarray(bundle.y_test).ravel()
    y_pred = np.asarray(best.y_pred).ravel()
    m = min(len(conf), len(y_true), len(y_pred))
    conf, y_true, y_pred = conf[:m], y_true[:m], y_pred[:m]
    correct = (y_pred == y_true).astype(int)

    # Calibration-style curve: accuracy & mean confidence per confidence bin.
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        bins.append({
            "bin": b,
            "mean_confidence": round(float(conf[sel].mean()), 4),
            "accuracy": round(float(correct[sel].mean()), 4),
            "n": int(sel.sum()),
        })

    # Flag rule: the least-confident `flag_frac` go to human review.
    thr = float(np.quantile(conf, flag_frac))
    flagged = conf <= thr
    n_err = int((1 - correct).sum())
    err_in_flagged = int(((1 - correct) & flagged).sum())

    auto = ~flagged
    return {
        "n_bins": len(bins),
        "bins": bins,
        "flag_fraction": flag_frac,
        "confidence_threshold": round(thr, 4),
        "coverage_flagged": round(float(flagged.mean()), 4),
        "accuracy_auto_accepted": round(float(correct[auto].mean()), 4) if np.any(auto) else None,
        "accuracy_flagged": round(float(correct[flagged].mean()), 4) if np.any(flagged) else None,
        "errors_total": n_err,
        "errors_caught_by_flag": err_in_flagged,
        "error_capture_rate": round(err_in_flagged / n_err, 4) if n_err else None,
        "overall_accuracy": round(float(correct.mean()), 4),
    }


def _reliability_note(rel: Optional[dict]) -> str:
    if not rel:
        return "reliability analysis unavailable."
    cap = rel.get("error_capture_rate")
    cov = rel.get("coverage_flagged")
    a_auto = rel.get("accuracy_auto_accepted")
    a_flag = rel.get("accuracy_flagged")
    if cap is None or a_auto is None:
        return "reliability curve computed."
    return (
        f"flagging the least-confident {cov*100:.0f}% of cases for review catches "
        f"{cap*100:.0f}% of all errors; the auto-accepted {100*(1-cov):.0f}% are "
        f"{a_auto*100:.1f}% accurate vs {a_flag*100:.1f}% on the flagged slice"
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _make_plots(shap_info: Optional[dict], rel: Optional[dict], cfg: Config) -> dict:
    figs: dict = {}
    try:
        from src import plots

        if shap_info and shap_info.get("global_importance"):
            names = [f for f, _ in shap_info["global_importance"]][::-1]
            vals = [v for _, v in shap_info["global_importance"]][::-1]
            figs["shap_importance"] = plots.plot_barh(
                names, vals, cfg.output_dir, "shap_importance",
                title="Global feature importance (mean |SHAP|)", xlabel="mean |SHAP value|",
            )
        if rel and rel.get("bins"):
            conf = [b["mean_confidence"] for b in rel["bins"]]
            acc = [b["accuracy"] for b in rel["bins"]]
            figs["reliability"] = plots.plot_reliability(
                conf, acc, cfg.output_dir, "reliability",
            )
    except Exception as exc:
        print(f"[stage4] plotting skipped: {type(exc).__name__}: {exc}")
    return figs


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(bundle: DataBundle, cfg: Config, ctx: Optional[dict] = None) -> dict:
    """Explain the best model and quantify when to trust it.

    Reads ``ctx['best_overall']`` (a ModelResult; its ``.estimator`` is the fitted
    pipeline and ``.y_proba`` the test probabilities). Produces a JSON-safe summary
    for the report; never raises (each piece degrades gracefully).
    """
    ctx = ctx if ctx is not None else {}
    best = ctx.get("best_overall")
    if best is None:
        print("[stage4] no best model in context — skipped.")
        return {"status": "skipped", "stage": "explain", "reason": "no best model"}

    print(f"[stage4] explaining '{best.name}' ...")
    shap_info = _shap_analysis(best, bundle, cfg)
    rel = _reliability(best, bundle)
    figs = _make_plots(shap_info, rel, cfg)

    if shap_info and shap_info.get("global_importance"):
        top3 = ", ".join(f for f, _ in shap_info["global_importance"][:3])
        print(f"[stage4] top drivers ({shap_info['method']}): {top3}")
    note = _reliability_note(rel)
    print(f"[stage4] {note}")

    return {
        "status": "ok",
        "stage": "explain",
        "model": best.name,
        "shap": shap_info,
        "reliability": rel,
        "reliability_note": note,
        "figures": figs,
    }
