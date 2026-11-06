"""Stage 3 — Hyperparameter-optimization comparison (the analytical centerpiece).

Tune the strongest booster four ways and record BOTH predictive score AND
computational cost (wall-clock seconds + number of model evaluations), to answer
the question the whole project is built around:

    "Does smarter search (Bayesian optimization) justify its cost, or is a
     default model — or a cheap random search — already good enough?"

Draws on Paper 2 (BO-XGBoost, ESWA 2026, DOI 10.1016/j.eswa.2026.131260): tuning a
gradient-boosting core with Bayesian optimization and comparing it against grid
search under a robust, time-aware cross-validation protocol.

    Default  ->  GridSearchCV  ->  RandomizedSearchCV  ->  Optuna (TPE / Bayesian)

Design decisions worth defending to an examiner
------------------------------------------------
* **What we tune.** ``ctx['best_booster_name']`` from Stage 2 (the single best
  booster, stacking excluded). The tool is general: a search space is defined for
  every booster it knows, so whichever wins Stage 2 can be tuned.

* **One CV protocol, shared by all.** All four strategies use the *same* CV object
  and the *same* scorer, so the only thing that varies is the search strategy.
  On time-ordered data that CV is ``TimeSeriesSplit``. This is safe here (unlike
  stacking): GridSearchCV/RandomizedSearchCV score folds with ``cross_validate``,
  not ``cross_val_predict``, so a non-partition splitter is fine.

* **Honest scoring.** Every candidate is ranked by cross-validated score on the
  *training* window only; the held-out temporal **test set is never touched during
  search**. The winning configuration of each strategy is then refit once and
  scored on that untouched test set through the same ``fit_evaluate`` path as every
  other model, so its row is directly comparable to Stages 1-2 on the leaderboard.

* **Equal-budget head-to-head.** Random and Bayesian each get exactly
  ``budget_trials`` candidate evaluations, so their comparison is apples-to-apples:
  same budget, only the sampling brain differs. Grid is exhaustive over a compact
  lattice and reported with its own (usually smaller, coarser) cost — it is the
  "no cleverness, just cover the corners" reference.

* **Fixed tree budget + single-threaded base learner.** All Stage-3 fits use a
  modest, fixed number of trees so (a) the default-vs-tuned gap reflects the tuned
  knobs rather than model size, and (b) a search of dozens of candidates stays
  laptop-friendly. Parallelism is applied at the *outer* level (folds/candidates
  run on all cores) while each base learner runs single-threaded — this avoids the
  cores-squared oversubscription of nesting ``n_jobs=-1`` inside ``n_jobs=-1``.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np

from .config import Config
from .data import DataBundle
from .metrics import higher_is_better, primary_metric_name
from .modeling import (
    ModelResult,
    fit_evaluate,
    leaderboard,
    log_result,
    make_pipeline,
    pick_best,
)


# --------------------------------------------------------------------------- #
# Search spaces — one per booster, expressed once and consumed three ways.
# Each entry gives a discrete ``grid`` (for GridSearchCV) plus a continuous
# ``low/high`` (for RandomizedSearchCV distributions and Optuna suggestions),
# so grid / random / Bayesian all explore the *same* region of the same knobs.
# --------------------------------------------------------------------------- #
_SEARCH_SPACES: dict[str, dict] = {
    "xgboost": {
        "max_depth": {"type": "int", "low": 3, "high": 9, "grid": [3, 6, 9]},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True,
                          "grid": [0.03, 0.1, 0.3]},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0, "grid": [0.7, 1.0]},
    },
    "lightgbm": {
        "num_leaves": {"type": "int", "low": 15, "high": 127, "grid": [15, 31, 63]},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True,
                          "grid": [0.03, 0.1, 0.3]},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0, "grid": [0.7, 1.0]},
    },
    "catboost": {
        "depth": {"type": "int", "low": 4, "high": 10, "grid": [4, 6, 8]},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True,
                          "grid": [0.03, 0.1, 0.3]},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True,
                        "grid": [1.0, 5.0]},
    },
    "random_forest": {
        "max_depth": {"type": "int", "low": 4, "high": 28, "grid": [8, 16, 24]},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20, "grid": [1, 5, 20]},
        "max_features": {"type": "cat", "choices": ["sqrt", "log2"],
                         "grid": ["sqrt", "log2"]},
    },
}

# Fixed, modest tree budget for every Stage-3 fit (speed + a fair default-vs-tuned gap).
_TUNE_TREES = 200
_TREE_PARAM = {
    "xgboost": "n_estimators",
    "lightgbm": "n_estimators",
    "catboost": "iterations",
    "random_forest": "n_estimators",
}
# Force the base learner single-threaded so outer (fold/candidate) parallelism owns the cores.
_SINGLE_THREAD = {
    "xgboost": {"n_jobs": 1},
    "lightgbm": {"n_jobs": 1, "verbose": -1},
    "catboost": {"thread_count": 1},
    "random_forest": {"n_jobs": 1},
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _scoring_for(bundle: DataBundle, cfg: Config) -> str:
    """The single sklearn scorer string shared by every strategy.

    sklearn scorers follow a "higher is better" convention (RMSE is negated), so
    the rest of this stage can always treat larger = better.
    """
    if cfg.tuning.scoring:
        return cfg.tuning.scoring
    if bundle.task == "classification":
        n = len(bundle.target_classes) if bundle.target_classes else 2
        return "roc_auc" if n == 2 else "roc_auc_ovr"
    return "neg_root_mean_squared_error"


def _make_cv(bundle: DataBundle, n_splits: int, rs: int):
    """Time-aware CV when the data is ordered, else a stratified/plain k-fold."""
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    if bundle.has_time:
        return TimeSeriesSplit(n_splits=n_splits)
    if bundle.task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    return KFold(n_splits=n_splits, shuffle=True, random_state=rs)


def _base_estimator(name: str, task: str, rs: int):
    """Fresh, unfitted booster with the Stage-3 tree cap + single-thread override."""
    from .boosting import _BUILDERS

    est = _BUILDERS[name](task, rs)
    overrides = {_TREE_PARAM[name]: _TUNE_TREES, **_SINGLE_THREAD.get(name, {})}
    try:
        est.set_params(**overrides)
    except Exception:
        pass  # if a param name ever drifts, fall back to builder defaults
    return est


def _grid(space: dict) -> tuple[dict, int]:
    """Pipeline-prefixed discrete grid + its exhaustive candidate count."""
    from sklearn.model_selection import ParameterGrid

    grid = {f"model__{k}": v["grid"] for k, v in space.items()}
    return grid, len(list(ParameterGrid(grid)))


def _distributions(space: dict) -> dict:
    """Pipeline-prefixed scipy distributions for RandomizedSearchCV."""
    from scipy.stats import loguniform, randint, uniform

    dist: dict = {}
    for k, v in space.items():
        key = f"model__{k}"
        if v["type"] == "int":
            dist[key] = randint(v["low"], v["high"] + 1)
        elif v["type"] == "float":
            dist[key] = (loguniform(v["low"], v["high"]) if v.get("log")
                         else uniform(v["low"], v["high"] - v["low"]))
        else:  # categorical
            dist[key] = list(v["choices"])
    return dist


def _best_from_cv_results(search) -> tuple[dict, float]:
    """Winning params (un-prefixed) + CV score, read straight from ``cv_results_``.

    We run the searches with ``refit=False`` (the winner is refit once, later,
    through ``fit_evaluate`` for a metrics-consistent test row), so we locate the
    best candidate ourselves. sklearn's scorer convention means argmax is always
    correct — even for negated RMSE.
    """
    res = search.cv_results_
    means = np.asarray(res["mean_test_score"], dtype=float)
    means = np.where(np.isnan(means), -np.inf, means)
    idx = int(np.argmax(means))
    params = {k.replace("model__", ""): v for k, v in res["params"][idx].items()}
    return params, float(res["mean_test_score"][idx])


def _evaluate_config(name: str, strategy: str, params: dict,
                     bundle: DataBundle, cfg: Config) -> ModelResult:
    """Refit one configuration on full train and score it on the untouched test set."""
    est = _base_estimator(name, bundle.task, cfg.random_state)
    if params:
        est.set_params(**params)
    return fit_evaluate(f"{name}_{strategy}", est, bundle, cfg)


# --------------------------------------------------------------------------- #
# The four strategies (each returns a JSON-safe record + a fitted ModelResult)
# --------------------------------------------------------------------------- #
def _run_default(name: str, bundle: DataBundle, cfg: Config) -> tuple[dict, ModelResult]:
    t0 = time.perf_counter()
    res = _evaluate_config(name, "default", {}, bundle, cfg)
    wall = time.perf_counter() - t0
    rec = {"strategy": "default", "best_params": {}, "cv_score": None,
           "n_candidates": 1, "n_evaluations": 1, "wall_seconds": round(wall, 2)}
    return rec, res


def _run_sklearn_search(kind: str, name: str, bundle: DataBundle, cfg: Config,
                        cv, scoring: str, n_splits: int,
                        budget: int) -> tuple[dict, ModelResult]:
    """GridSearchCV (exhaustive) or RandomizedSearchCV (``budget`` samples)."""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    space = _SEARCH_SPACES[name]
    base = _base_estimator(name, bundle.task, cfg.random_state)
    pipe = make_pipeline(bundle.preprocessor, base)

    if kind == "grid":
        grid, n_cand = _grid(space)
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1,
                              refit=False, error_score=np.nan)
    else:  # random
        n_cand = budget
        search = RandomizedSearchCV(
            pipe, _distributions(space), n_iter=n_cand, scoring=scoring, cv=cv,
            n_jobs=-1, refit=False, random_state=cfg.random_state, error_score=np.nan,
        )

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(bundle.X_train, bundle.y_train)
    wall = time.perf_counter() - t0

    params, cv_score = _best_from_cv_results(search)
    res = _evaluate_config(name, kind, params, bundle, cfg)
    rec = {"strategy": kind, "best_params": params, "cv_score": round(cv_score, 5),
           "n_candidates": int(n_cand), "n_evaluations": int(n_cand * n_splits),
           "wall_seconds": round(wall, 2)}
    return rec, res


def _run_optuna(name: str, bundle: DataBundle, cfg: Config, cv, scoring: str,
                n_splits: int, budget: int) -> tuple[dict, ModelResult]:
    """Bayesian optimization (Optuna TPE) — ``budget`` trials, same CV & scorer."""
    import optuna
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = _SEARCH_SPACES[name]
    rs = cfg.random_state

    def objective(trial) -> float:
        params = {}
        for k, v in space.items():
            if v["type"] == "int":
                params[f"model__{k}"] = trial.suggest_int(k, v["low"], v["high"])
            elif v["type"] == "float":
                params[f"model__{k}"] = trial.suggest_float(
                    k, v["low"], v["high"], log=bool(v.get("log", False))
                )
            else:
                params[f"model__{k}"] = trial.suggest_categorical(k, v["choices"])

        pipe = make_pipeline(bundle.preprocessor, _base_estimator(name, bundle.task, rs))
        pipe.set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, bundle.X_train, bundle.y_train,
                                     cv=cv, scoring=scoring, n_jobs=-1,
                                     error_score=np.nan)
        return float(np.nanmean(scores)) if np.any(~np.isnan(scores)) else -1e9

    sampler = optuna.samplers.TPESampler(seed=rs)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    wall = time.perf_counter() - t0

    params = dict(study.best_params)  # already un-prefixed (bare knob names)
    res = _evaluate_config(name, "bayes", params, bundle, cfg)
    rec = {"strategy": "bayes", "best_params": params,
           "cv_score": round(float(study.best_value), 5),
           "n_candidates": int(budget), "n_evaluations": int(budget * n_splits),
           "wall_seconds": round(wall, 2)}
    return rec, res


# --------------------------------------------------------------------------- #
# Verdict — the plain-English "was tuning worth it?" takeaway
# --------------------------------------------------------------------------- #
def _verdict(records: list[dict], results: list[ModelResult], task: str) -> str:
    metric = primary_metric_name(task)
    by_strategy = {r.name.split("_")[-1]: r for r in results}
    parts = []

    hib = higher_is_better(metric)

    default = by_strategy.get("default")
    best = pick_best(results, task)
    if default is not None and best is not None:
        d = default.metrics.get(metric, float("nan"))
        b = best.metrics.get(metric, float("nan"))
        # Report the improvement as a magnitude in the metric's own direction, so
        # an rmse that dropped reads as a gain rather than a negative number.
        delta = (b - d) if hib else (d - b)
        won = best.name.split("_")[-1]
        if won == "default":
            parts.append(
                f"tuning did NOT beat the default ({metric}={d:.4f}); "
                f"on this data a sensible default is already competitive"
            )
        else:
            parts.append(
                f"best tuner '{won}' reached {metric}={b:.4f} vs default {d:.4f} "
                f"(+{delta:.4f})"
            )

    # Equal-budget head-to-head: Bayesian vs Random (identical trial count).
    rec_by = {r["strategy"]: r for r in records}
    bayes, rand = by_strategy.get("bayes"), by_strategy.get("random")
    if bayes is not None and rand is not None:
        bm = bayes.metrics.get(metric, float("nan"))
        rm = rand.metrics.get(metric, float("nan"))
        bt = rec_by.get("bayes", {}).get("wall_seconds")
        rt = rec_by.get("random", {}).get("wall_seconds")
        # Direction matters: rmse/mae are error metrics, so smaller wins.
        bayes_won = (bm >= rm) if hib else (bm <= rm)
        better = "Bayesian" if bayes_won else "random"
        parts.append(
            f"at equal budget, {better} search won ({metric}: bayes {bm:.4f} @ {bt}s "
            f"vs random {rm:.4f} @ {rt}s)"
        )
    return "; ".join(parts) if parts else "tuning comparison complete."


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(bundle: DataBundle, cfg: Config, ctx: Optional[dict] = None) -> dict:
    """Compare Default / Grid / Random / Bayesian on the best booster.

    Reads ``ctx['best_booster_name']`` (Stage 2). Registers every tuned result in
    ``ctx['results']``, records ``ctx['tuned_best']``, and — if a tuned model is now
    the best across all stages — updates ``ctx['best_overall']`` / ``ctx['best_model']``
    so Stages 4-5 explain and monitor the genuinely strongest model.
    """
    ctx = ctx if ctx is not None else {}
    name = ctx.get("best_booster_name")

    if not name or name not in _SEARCH_SPACES:
        print(f"[stage3] no tunable booster available (best_booster_name={name!r}) — skipped.")
        return {"status": "skipped", "stage": "tuning", "reason": "no tunable booster",
                "best_booster_name": name}

    task = bundle.task
    rs = cfg.random_state
    n_splits = cfg.tuning.cv_splits
    budget = cfg.tuning.budget_trials
    scoring = _scoring_for(bundle, cfg)
    cv = _make_cv(bundle, n_splits, rs)

    print(f"[stage3] tuning '{name}' | scoring={scoring} | cv={n_splits}-fold"
          f"{' (TimeSeriesSplit)' if bundle.has_time else ''} | budget={budget} trials")

    plan = [
        ("default", lambda: _run_default(name, bundle, cfg)),
        ("grid", lambda: _run_sklearn_search("grid", name, bundle, cfg, cv, scoring, n_splits, budget)),
        ("random", lambda: _run_sklearn_search("random", name, bundle, cfg, cv, scoring, n_splits, budget)),
        ("bayes", lambda: _run_optuna(name, bundle, cfg, cv, scoring, n_splits, budget)),
    ]

    records: list[dict] = []
    results: list[ModelResult] = []
    for strat, fn in plan:
        try:
            rec, res = fn()
            records.append(rec)
            results.append(res)
            log_result("stage3", res, task, cfg.report.fbeta)
            extra = ""
            if rec.get("cv_score") is not None:
                extra = f"  cv={rec['cv_score']:.4f}  evals={rec['n_evaluations']}  ({rec['wall_seconds']}s)"
            print(f"[stage3]   ^ {strat}: {rec.get('best_params', {})}{extra}")
        except ImportError as exc:
            print(f"[stage3] {strat} unavailable (library not installed): {exc} — skipped.")
        except Exception as exc:
            print(f"[stage3] {strat} failed: {type(exc).__name__}: {exc} — skipped.")

    if not results:
        return {"status": "error", "stage": "tuning", "booster": name,
                "reason": "every tuning strategy failed"}

    # Attach cost accounting to each ModelResult for the report, then register.
    reg = ctx.setdefault("results", {})
    for rec, res in zip(records, results):
        res.extra["tuning"] = {k: rec[k] for k in ("strategy", "cv_score",
                               "n_evaluations", "wall_seconds", "best_params")}
        reg[res.name] = res

    tuned_best = pick_best(results, task)
    ctx["tuned_best"] = tuned_best

    # Promote to overall-best if a tuned model now leads the whole leaderboard.
    overall = pick_best(list(reg.values()), task)
    if overall is not None:
        ctx["best_overall"] = overall
        ctx["best_model"] = overall.estimator

    verdict = _verdict(records, results, task)
    print(f"[stage3] verdict: {verdict}")

    return {
        "status": "ok",
        "stage": "tuning",
        "booster": name,
        "scoring": scoring,
        "cv_splits": n_splits,
        "primary_metric": primary_metric_name(task),
        "budget_trials": budget,
        "records": records,
        "best_strategy": tuned_best.name.split("_")[-1] if tuned_best else None,
        "tuned_best": tuned_best.name if tuned_best else None,
        "leaderboard": leaderboard(list(reg.values()), task),
        "verdict": verdict,
    }
