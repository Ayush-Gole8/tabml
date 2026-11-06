"""Stage 1 — Regularized baselines (the bias-variance foundation).

Before any heavy model, establish an honest, interpretable yardstick: linear
models under different regularizers. This makes two teaching points concrete and
measurable:

  * **L2 (ridge-style) shrinks** coefficients smoothly toward zero;
  * **L1 (lasso-style) zeroes** coefficients outright -> built-in feature
    selection (we report the resulting sparsity);
  * **ElasticNet** blends the two.

The classification analog of Ridge/Lasso/ElasticNet is Logistic Regression with
``penalty in {l2, l1, elasticnet}`` — identical regularization mathematics on the
linear term, just wrapped in a logistic loss. (Be ready to defend this to an
examiner: "Ridge/Lasso are regression names; the *penalty* is the idea, and it
transfers directly to logistic regression.")

Draws on Paper 1 (RegStack, ESWA 2025, DOI 10.1016/j.eswa.2025.127766): treating
regularization as a first-class ingredient, not an afterthought.

    Classification -> LogisticRegression(penalty in {None, l2, l1, elasticnet})
    Regression     -> LinearRegression / Ridge / Lasso / ElasticNet
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .config import Config
from .data import DataBundle
from .metrics import primary_metric_name
from .modeling import (
    ModelResult,
    feature_names_out,
    fit_evaluate,
    leaderboard,
    log_result,
    pick_best,
)


# --------------------------------------------------------------------------- #
# Model zoo (built lazily so an import problem surfaces as a clear message)
# --------------------------------------------------------------------------- #
def _sklearn_new_api() -> bool:
    """True on scikit-learn >= 1.8, where ``penalty=`` is deprecated for ``l1_ratio``.

    On 1.8+ a stray ``penalty=`` is silently overridden by ``l1_ratio``'s default
    of 0.0 (pure L2) — which would make all four regularizers identical L2 fits.
    Detecting the version lets us speak the right dialect.
    """
    import sklearn

    try:
        major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
        return (major, minor) >= (1, 8)
    except Exception:
        return False


def _make_logreg(kind: str, C, rs: int, new_api: bool):
    """LogisticRegression for kind in {none, l2, l1, elasticnet} at strength ``C``.

    On 1.8+ the elasticnet spectrum is expressed via ``l1_ratio`` (0=L2, 1=L1,
    0.5=EN) and ``C=inf`` means unregularized; 'saga' is required once
    ``l1_ratio > 0`` (lbfgs is fine and faster at 0). ``n_jobs`` is omitted: it has
    no effect on LogisticRegression and was removed in 1.8.
    """
    from sklearn.linear_model import LogisticRegression

    if new_api:
        cfgs = {
            "none": dict(C=np.inf, l1_ratio=0.0, solver="lbfgs", max_iter=3000),
            "l2": dict(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=3000),
            "l1": dict(C=C, l1_ratio=1.0, solver="saga", max_iter=5000, tol=1e-3, random_state=rs),
            "elasticnet": dict(C=C, l1_ratio=0.5, solver="saga", max_iter=5000, tol=1e-3, random_state=rs),
        }
    else:
        cfgs = {
            "none": dict(penalty=None, solver="lbfgs", max_iter=3000),
            "l2": dict(penalty="l2", C=C, solver="lbfgs", max_iter=3000),
            "l1": dict(penalty="l1", C=C, solver="saga", max_iter=5000, tol=1e-3, random_state=rs),
            "elasticnet": dict(penalty="elasticnet", C=C, l1_ratio=0.5, solver="saga",
                               max_iter=5000, tol=1e-3, random_state=rs),
        }
    return LogisticRegression(**cfgs[kind])


def _classification_models(cfg: Config) -> dict:
    """Four logistic-regression regularizers at a common C=1.0: none / L2 / L1 / EN.

    Keeping C fixed across penalties makes the comparison fair. The
    *regularization path* (below) is what reveals what each penalty actually does
    as its strength varies — the headline four can look near-identical when the
    data is abundant and low-dimensional, which is itself a finding.
    """
    rs = cfg.random_state
    new_api = _sklearn_new_api()
    return {
        "logreg_none": _make_logreg("none", None, rs, new_api),
        "logreg_l2": _make_logreg("l2", 1.0, rs, new_api),
        "logreg_l1": _make_logreg("l1", 1.0, rs, new_api),
        "logreg_elasticnet": _make_logreg("elasticnet", 1.0, rs, new_api),
    }


def _regression_models(cfg: Config) -> dict:
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

    rs = cfg.random_state
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=rs),
        "lasso": Lasso(alpha=0.001, max_iter=10000, random_state=rs),
        "elasticnet": ElasticNet(
            alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=rs
        ),
    }


# --------------------------------------------------------------------------- #
# Coefficient diagnostics — the "regularization actually did something" evidence
# --------------------------------------------------------------------------- #
def _coef_diagnostics(res: ModelResult) -> dict:
    """Sparsity, coefficient norms, and the strongest features for one model."""
    model = res.estimator.named_steps["model"]
    coef = getattr(model, "coef_", None)
    if coef is None:
        return {}
    coef = np.asarray(coef, dtype=float)
    names = feature_names_out(res.estimator)

    flat = coef.ravel()
    n_total = int(flat.size)
    n_zero = int(np.sum(np.abs(flat) < 1e-6))

    # Rank features by importance. For multiclass, average |coef| across classes
    # so each feature gets one score aligned with the feature-name list.
    if coef.ndim > 1 and coef.shape[0] > 1:
        strength = np.abs(coef).mean(axis=0)
    else:
        strength = np.abs(flat)
    order = np.argsort(strength)[::-1][:12]
    signed = flat if flat.size == strength.size else coef.mean(axis=0)
    top = [
        [names[i] if i < len(names) else f"f{i}", round(float(signed[i]), 5)]
        for i in order
    ]

    return {
        "n_coefficients": n_total,
        "n_zero": n_zero,
        "sparsity": round(n_zero / n_total, 4) if n_total else 0.0,
        "l1_norm": round(float(np.abs(flat).sum()), 5),
        "l2_norm": round(float(np.linalg.norm(flat)), 5),
        "top_features": top,
    }


def _regularization_note(results: list[ModelResult], task: str, path: Optional[dict] = None) -> str:
    """A plain-English takeaway: why the headline looks flat, and what the path shows."""
    metric = primary_metric_name(task)
    vals = [r.metrics.get(metric, float("nan")) for r in results]
    vals = [v for v in vals if v == v]  # drop nan
    best = pick_best(results, task)

    parts = []
    if best is not None and vals:
        spread = max(vals) - min(vals)
        parts.append(
            f"the {len(results)} regularizers span only {spread:.4f} {metric} "
            f"(best '{best.name}' {best.metrics.get(metric, float('nan')):.4f}) — "
            f"with abundant rows and few features, penalties barely move the fit at the default strength"
        )

    if path:
        F = path.get("n_features", 0)
        if path.get("type") == "classification" and path.get("l1", {}).get("nonzero"):
            nz, Cs = path["l1"]["nonzero"], path["C"]
            i = int(np.argmin(nz))
            parts.append(
                f"L1 path: at C={Cs[i]:g} only {nz[i]}/{F} features survive (sparsity), "
                f"rising to {max(nz)}/{F} as C grows"
            )
        elif path.get("type") == "regression" and path.get("lasso", {}).get("nonzero"):
            nz, al = path["lasso"]["nonzero"], path["alpha"]
            i = int(np.argmax(al))  # strongest penalty = largest alpha
            parts.append(
                f"Lasso path: at alpha={al[i]:g} only {nz[i]}/{F} features survive (sparsity)"
            )

    return "; ".join(parts) if parts else "regularized linear baselines fitted."


# --------------------------------------------------------------------------- #
# Regularization path — the "what does the penalty actually do?" demonstration
# --------------------------------------------------------------------------- #
def _regularization_path(bundle: DataBundle, cfg: Config) -> Optional[dict]:
    """Sweep penalty strength to visualize L1 sparsity and L2 shrinkage.

    Fits on an INTERNAL train/validation split of the *training* data only (the
    held-out test set is never touched), so this is a clean diagnostic rather than
    test-set peeking. Linear fits are cheap, so a 12-point sweep costs seconds.

    Returns per-strength: number of surviving (non-zero) coefficients, the L2 norm
    of the coefficient vector (shrinkage), and a validation score (the bias-variance
    sweet spot). Classification sweeps C for L1 & L2; regression sweeps alpha for
    Lasso & Ridge.
    """
    try:
        from .modeling import make_pipeline

        rs = cfg.random_state
        n = len(bundle.X_train)
        if n < 50:
            return None
        cut = int(n * 0.8)
        Xtr, Xval = bundle.X_train.iloc[:cut], bundle.X_train.iloc[cut:]
        ytr, yval = bundle.y_train[:cut], bundle.y_train[cut:]

        def _fit_coef(estimator):
            pipe = make_pipeline(bundle.preprocessor, estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(Xtr, ytr)
            coef = np.asarray(pipe.named_steps["model"].coef_, dtype=float).ravel()
            return pipe, coef

        if bundle.task == "classification":
            from sklearn.metrics import roc_auc_score

            new_api = _sklearn_new_api()
            Cs = np.logspace(-3, 2, 12)
            out: dict = {
                "type": "classification",
                "C": [round(float(c), 6) for c in Cs],
                "l1": {"nonzero": [], "l2norm": [], "val_roc_auc": []},
                "l2": {"nonzero": [], "l2norm": [], "val_roc_auc": []},
            }
            n_features = 0
            for kind in ("l1", "l2"):
                for c in Cs:
                    pipe, coef = _fit_coef(_make_logreg(kind, float(c), rs, new_api))
                    n_features = coef.size
                    out[kind]["nonzero"].append(int(np.sum(np.abs(coef) > 1e-6)))
                    out[kind]["l2norm"].append(round(float(np.linalg.norm(coef)), 5))
                    try:
                        proba = pipe.predict_proba(Xval)[:, 1]
                        out[kind]["val_roc_auc"].append(round(float(roc_auc_score(yval, proba)), 4))
                    except Exception:
                        out[kind]["val_roc_auc"].append(float("nan"))
            out["n_features"] = int(n_features)
            return out

        # regression
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.metrics import mean_squared_error

        alphas = np.logspace(-4, 1, 12)
        out = {
            "type": "regression",
            "alpha": [round(float(a), 6) for a in alphas],
            "lasso": {"nonzero": [], "l2norm": [], "val_rmse": []},
            "ridge": {"nonzero": [], "l2norm": [], "val_rmse": []},
        }
        n_features = 0
        for kind in ("lasso", "ridge"):
            for a in alphas:
                est = (
                    Lasso(alpha=float(a), max_iter=10000, random_state=rs)
                    if kind == "lasso"
                    else Ridge(alpha=float(a), random_state=rs)
                )
                pipe, coef = _fit_coef(est)
                n_features = coef.size
                out[kind]["nonzero"].append(int(np.sum(np.abs(coef) > 1e-6)))
                out[kind]["l2norm"].append(round(float(np.linalg.norm(coef)), 5))
                pred = pipe.predict(Xval)
                out[kind]["val_rmse"].append(round(float(np.sqrt(mean_squared_error(yval, pred))), 5))
        out["n_features"] = int(n_features)
        return out
    except Exception as exc:  # a diagnostic must never sink the pipeline
        print(f"[stage1] regularization path skipped: {type(exc).__name__}: {exc}")
        return None


def _print_path(path: Optional[dict]) -> None:
    """Compact console view of the sweep so the mechanism is visible during a run."""
    if not path:
        return
    F = path.get("n_features", 0)
    if path.get("type") == "classification":
        Cs = path["C"]
        l1nz, l2n = path["l1"]["nonzero"], path["l2"]["l2norm"]
        lo, hi = 0, len(Cs) - 1
        print(
            f"[stage1] reg-path (of {F} features): "
            f"L1 keeps {l1nz[lo]}@C={Cs[lo]:g} -> {l1nz[hi]}@C={Cs[hi]:g} (sparsity); "
            f"L2 |coef| grows {l2n[lo]:g} -> {l2n[hi]:g} (shrinkage)"
        )
    elif path.get("type") == "regression":
        al = path["alpha"]
        lnz = path["lasso"]["nonzero"]
        lo, hi = 0, len(al) - 1
        print(
            f"[stage1] reg-path (of {F} features): "
            f"Lasso keeps {lnz[hi]}@alpha={al[hi]:g} -> {lnz[lo]}@alpha={al[lo]:g} (sparsity)"
        )


# --------------------------------------------------------------------------- #
# Learning curve — bias/variance on the strongest baseline only (cost-bounded)
# --------------------------------------------------------------------------- #
def _learning_curve(best: ModelResult, bundle: DataBundle, cfg: Config) -> Optional[dict]:
    """Train/validation score vs. training-set size, for the best baseline.

    Uses the pipeline (so preprocessing is refit inside each fold — no leakage),
    a time-aware CV when we have a time column, and only 5 points on 3 folds to
    stay cheap. ``learning_curve`` uses cross_validate internally, so
    ``TimeSeriesSplit`` is fine here (unlike stacking's cross_val_predict).
    """
    try:
        from sklearn.base import clone
        from sklearn.model_selection import (
            KFold,
            StratifiedKFold,
            TimeSeriesSplit,
            learning_curve,
        )

        n_classes = len(bundle.target_classes) if bundle.target_classes else 2
        if bundle.task == "classification":
            scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
        else:
            scoring = "neg_root_mean_squared_error"

        if bundle.has_time:
            cv = TimeSeriesSplit(n_splits=3)
        elif bundle.task == "classification":
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.random_state)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=cfg.random_state)

        sizes, train_scores, val_scores = learning_curve(
            clone(best.estimator),
            bundle.X_train,
            bundle.y_train,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            shuffle=False,
        )
        return {
            "model": best.name,
            "metric": scoring,
            "train_sizes": [int(s) for s in sizes],
            "train_mean": [round(float(v), 4) for v in train_scores.mean(axis=1)],
            "val_mean": [round(float(v), 4) for v in val_scores.mean(axis=1)],
        }
    except Exception as exc:  # never let a diagnostic sink the pipeline
        print(f"[stage1] learning curve skipped: {type(exc).__name__}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(bundle: DataBundle, cfg: Config, ctx: Optional[dict] = None) -> dict:
    """Fit regularized baselines; return metrics + coefficient diagnostics.

    Side effect: registers every fitted result in ``ctx['results']`` and records
    ``ctx['baseline_best']`` so later stages (and Stage 2's overall-best pick) can
    see the baselines.
    """
    ctx = ctx if ctx is not None else {}
    task = bundle.task
    models = _classification_models(cfg) if task == "classification" else _regression_models(cfg)

    results: list[ModelResult] = []
    for name, est in models.items():
        try:
            res = fit_evaluate(name, est, bundle, cfg)
            res.extra["coefficients"] = _coef_diagnostics(res)
            results.append(res)
            log_result("stage1", res, task, cfg.report.fbeta)
        except Exception as exc:
            print(f"[stage1] {name} failed: {type(exc).__name__}: {exc} — skipped.")

    if not results:
        return {"status": "error", "stage": "baselines", "reason": "no baseline trained"}

    best = pick_best(results, task)
    reg = ctx.setdefault("results", {})
    for r in results:
        reg[r.name] = r
    ctx["baseline_best"] = best

    lc = _learning_curve(best, bundle, cfg)
    path = _regularization_path(bundle, cfg)
    _print_path(path)
    note = _regularization_note(results, task, path)

    return {
        "status": "ok",
        "stage": "baselines",
        "primary_metric": primary_metric_name(task),
        "best": best.name,
        "models": [r.summary() for r in results],
        "leaderboard": leaderboard(results, task),
        "learning_curve": lc,
        "regularization_path": path,
        "note": note,
    }
