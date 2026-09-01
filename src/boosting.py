"""Stage 2 — Boosting ensembles + a stacking meta-model (the workhorses).

Three gradient-boosting libraries (XGBoost, LightGBM, CatBoost), a RandomForest
as an older-ensemble reference point, and a stacking model whose meta-learner is
a regularized linear model — RegStack in spirit.

Draws on Paper 1 (RegStack, ESWA 2025 — regularized stacking) and Paper 2
(BO-XGBoost, ESWA 2026, DOI 10.1016/j.eswa.2026.131260 — XGBoost as the core
learner, later tuned in Stage 3).

Robustness choices worth knowing:
  * **Optional dependencies.** Each booster is imported lazily; if a library
    isn't installed (or fails to construct), that model is skipped with a message
    and the pipeline carries on. The tool degrades gracefully instead of dying.
  * **Stacking CV must be a partition.** scikit-learn's StackingClassifier builds
    meta-features with ``cross_val_predict``, which requires every sample to fall
    in exactly one validation fold. ``TimeSeriesSplit`` is *not* a partition
    (its earliest rows never appear in a validation fold) and would raise
    "cross_val_predict only works for partitions". So the meta-feature CV uses
    stratified/plain k-fold. Time-honesty is preserved because the held-out
    *temporal* test set is never part of this CV and is only scored at the end.
  * **No nested oversubscription.** Base learners use all cores (n_jobs=-1);
    stacking itself runs folds sequentially (n_jobs=1) so we don't spawn
    cores^2 threads on a laptop.
  * **Uniform preprocessing.** Trees don't need scaling/one-hot, but every model
    shares the one leakage-safe preprocessor so the comparison is clean and the
    system stays simple. (Scaling is harmless to trees.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import Config
from src.data import DataBundle
from src.metrics import primary_metric_name
from src.modeling import (
    ModelResult,
    feature_names_out,
    fit_evaluate,
    leaderboard,
    log_result,
    pick_best,
)


# --------------------------------------------------------------------------- #
# Model builders (lazy imports: a missing library skips only that model)
# --------------------------------------------------------------------------- #
def _build_xgboost(task: str, rs: int):
    import xgboost as xgb  # noqa: F401  (import may raise ImportError -> skipped)

    params = dict(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist",          # histogram splits -> scales to large data
        random_state=rs, n_jobs=-1,
    )
    if task == "classification":
        return xgb.XGBClassifier(eval_metric="logloss", **params)
    return xgb.XGBRegressor(**params)


def _build_lightgbm(task: str, rs: int):
    import lightgbm as lgb

    params = dict(
        n_estimators=300, learning_rate=0.1, num_leaves=31, max_depth=-1,
        subsample=0.9, colsample_bytree=0.9,
        random_state=rs, n_jobs=-1, verbose=-1,   # verbose=-1 silences LightGBM
    )
    if task == "classification":
        return lgb.LGBMClassifier(**params)
    return lgb.LGBMRegressor(**params)


def _build_catboost(task: str, rs: int):
    from catboost import CatBoostClassifier, CatBoostRegressor

    params = dict(
        iterations=300, learning_rate=0.1, depth=6, random_seed=rs,
        verbose=0,
        allow_writing_files=False,   # don't litter a catboost_info/ dir (Windows!)
        thread_count=-1,
    )
    if task == "classification":
        return CatBoostClassifier(**params)
    return CatBoostRegressor(**params)


def _build_random_forest(task: str, rs: int):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    params = dict(n_estimators=300, n_jobs=-1, random_state=rs)
    if task == "classification":
        return RandomForestClassifier(**params)
    return RandomForestRegressor(**params)


_BUILDERS = {
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
    "catboost": _build_catboost,
    "random_forest": _build_random_forest,
}


# --------------------------------------------------------------------------- #
# Stacking meta-model
# --------------------------------------------------------------------------- #
def _build_stacking(task: str, available: list[str], rs: int, cv_splits: int):
    """A stacking ensemble over the available boosters with a regularized meta-learner.

    See the module docstring for why the internal CV is k-fold, not TimeSeriesSplit.
    """
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import KFold, StratifiedKFold

    # Fresh, unfitted base estimators (never reuse ones already fitted above).
    estimators = [(name, _BUILDERS[name](task, rs)) for name in available]

    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=rs)
        # Regularized (L2 by default) linear meta-learner. No penalty= arg: it is
        # the default on old sklearn and is deprecated on 1.8+ (would warn).
        final = LogisticRegression(max_iter=2000)
        return StackingClassifier(
            estimators=estimators, final_estimator=final, cv=cv,
            n_jobs=1, passthrough=False,
        )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
    final = RidgeCV()  # regularized linear meta-learner for regression
    return StackingRegressor(
        estimators=estimators, final_estimator=final, cv=cv,
        n_jobs=1, passthrough=False,
    )


# --------------------------------------------------------------------------- #
# Feature-importance diagnostics (tree models expose feature_importances_)
# --------------------------------------------------------------------------- #
def _importance_diagnostics(res: ModelResult) -> dict:
    model = res.estimator.named_steps["model"]
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    names = feature_names_out(res.estimator)
    order = np.argsort(imp)[::-1][:15]
    top = [
        [names[i] if i < len(names) else f"f{i}", round(float(imp[i]), 5)]
        for i in order
    ]
    return {"top_features": top}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(bundle: DataBundle, cfg: Config, ctx: Optional[dict] = None) -> dict:
    """Train ensembles + stacking; return a leaderboard and stash the best models.

    Side effects on ``ctx``:
      * ``ctx['results']``       — every fitted ModelResult, keyed by name
      * ``ctx['ensemble_best']`` — best Stage-2 model
      * ``ctx['best_booster']``  / ``ctx['best_booster_name']`` — best *single*
        booster (excludes stacking) -> what Stage 3 will tune
      * ``ctx['best_overall']``  / ``ctx['best_model']`` — best across ALL stages
        so far (baselines + ensembles) -> what Stages 4/5 explain & monitor
    """
    ctx = ctx if ctx is not None else {}
    task = bundle.task
    rs = cfg.random_state
    requested = list(cfg.models.boosters)

    results: list[ModelResult] = []
    available: list[str] = []

    for name in requested:
        builder = _BUILDERS.get(name)
        if builder is None:
            print(f"[stage2] unknown booster '{name}' — skipped.")
            continue
        try:
            est = builder(task, rs)
        except ImportError as exc:
            print(f"[stage2] '{name}' unavailable (library not installed): {exc} — skipped.")
            continue
        except Exception as exc:
            print(f"[stage2] '{name}' could not be constructed: {exc} — skipped.")
            continue
        try:
            res = fit_evaluate(name, est, bundle, cfg)
            res.extra["importance"] = _importance_diagnostics(res)
            results.append(res)
            available.append(name)
            log_result("stage2", res, task, cfg.report.fbeta)
        except Exception as exc:
            print(f"[stage2] '{name}' failed during fit/eval: {type(exc).__name__}: {exc} — skipped.")

    # Stacking needs at least two working base learners to be meaningful.
    if cfg.models.stacking:
        if len(available) >= 2:
            try:
                stack = _build_stacking(task, available, rs, cfg.tuning.cv_splits)
                res = fit_evaluate("stacking", stack, bundle, cfg)
                results.append(res)
                log_result("stage2", res, task, cfg.report.fbeta)
            except Exception as exc:
                print(f"[stage2] stacking failed: {type(exc).__name__}: {exc} — skipped.")
        else:
            print(f"[stage2] stacking skipped: needs >=2 base learners, have {len(available)}.")

    if not results:
        # Every booster was skipped (e.g. none of the libraries are installed).
        # Don't starve downstream stages: fall back to the best baseline that
        # Stage 1 already registered in the shared context.
        reg = ctx.setdefault("results", {})
        fallback = pick_best(list(reg.values()), task) if reg else None
        if fallback is not None:
            ctx["best_overall"] = fallback
            ctx["best_model"] = fallback.estimator
        return {
            "status": "error",
            "stage": "boosting",
            "reason": "no ensemble model trained — are xgboost/lightgbm/catboost installed?",
            "requested": requested,
            "fell_back_to": fallback.name if fallback else None,
        }

    # Register in shared context.
    reg = ctx.setdefault("results", {})
    for r in results:
        reg[r.name] = r

    best_ensemble = pick_best(results, task)
    single = [r for r in results if r.name in _BUILDERS]     # exclude stacking
    best_booster = pick_best(single, task) if single else best_ensemble
    best_overall = pick_best(list(reg.values()), task)        # incl. baselines

    ctx["ensemble_best"] = best_ensemble
    ctx["best_booster"] = best_booster
    ctx["best_booster_name"] = best_booster.name if best_booster else None
    ctx["best_overall"] = best_overall
    ctx["best_model"] = best_overall.estimator if best_overall else None

    return {
        "status": "ok",
        "stage": "boosting",
        "primary_metric": primary_metric_name(task),
        "available_boosters": available,
        "skipped_boosters": [n for n in requested if n not in available],
        "best": best_ensemble.name if best_ensemble else None,
        "best_booster": best_booster.name if best_booster else None,
        "best_overall": best_overall.name if best_overall else None,
        "models": [r.summary() for r in results],
        "leaderboard": leaderboard(list(reg.values()), task),
    }
