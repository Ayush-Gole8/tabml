# TabML-Guard — Evaluation / Viva Preparation Guide

*Everything you need to explain this project tomorrow: the plain-English story
first, then the technical implementation in full detail, then the five Q1
references, the syllabus mapping, and a bank of likely examiner questions with
crisp answers.*

---

## Part 0 — The 60-second version (say this first)

**Project title:** *Explainable and Drift-Aware Hyperparameter Optimization of
Ensemble Machine Learning Models for Real-World Tabular Prediction.*

**What we built:** a single, reusable, config-driven pipeline called
**TabML-Guard**. You point it at any time-ordered tabular dataset by editing one
config file, and it automatically runs the whole modern supervised-learning
workflow end to end — regularized baselines → boosting ensembles + stacking →
a Grid/Random/Bayesian hyperparameter-tuning *comparison* → SHAP explainability +
a reliability layer → concept-drift monitoring with statistical significance
tests — and emits a single self-contained **HTML report** with a plain-English
verdict on top.

**The framing that matters:** this is an *implementation* project, not a new
algorithm. The contribution is the **reusable system** and the **empirical
study** it produces. The electricity-price dataset (ELEC2) is just the demo case;
the same code runs on any dataset by config alone.

**One-line decode of the title:**

| Phrase in title | What it means in our code |
|---|---|
| *Ensemble ML models* | XGBoost, LightGBM, CatBoost, RandomForest + a stacking meta-model |
| *Hyperparameter Optimization* | Default vs Grid vs Random vs Bayesian (Optuna), scored on **accuracy AND cost** |
| *Explainable* | SHAP global + local, plus a confidence-based reliability/triage layer |
| *Drift-Aware* | Walk-forward prequential monitoring, retraining, PSI, McNemar/Wilcoxon tests |
| *Real-World Tabular Prediction* | Time-ordered tabular data; honest temporal train/test split |

---

## Part 1 — The project in plain English

### The problem we're solving

Most student ML projects train one model on one dataset, print an accuracy
number, and stop. Three things are missing that matter in the real world:

1. **Is the fancy model actually worth it?** People reach for XGBoost and expensive
   tuning by reflex. Sometimes a simple regularized linear model is just as good,
   and sometimes an untuned default already matches an expensive Bayesian search.
   Nobody checks. We check, and we measure the *cost* of tuning, not just the gain.

2. **Can you trust an individual prediction?** An accuracy of 88% means 12% are
   wrong — but *which* 12%? We use the model's own confidence to flag the shaky
   predictions for human review, and we prove that the flagged pile really does
   contain most of the errors.

3. **Does the model rot over time?** A model trained on the past can quietly stop
   working as the world changes ("concept drift"). We replay the test data in
   time order, watch performance chunk by chunk, and race a frozen model against
   one that keeps retraining.

### The analogy

Think of TabML-Guard as a **diagnostic garage for a prediction problem**. You
drive in any dataset. The garage runs a fixed, trusted battery of tests —
compression check, brakes, emissions — and hands you back a **health report** with
a big verdict at the top ("recommended model: X; here's the one thing you should
know") and the detailed readouts below. You don't rewire the garage for each car;
you just drive a different car in. Here, "driving a different car in" = editing
the config file.

### The headline deliverable: the HTML report

The single most important output is `results/report_<dataset>.html`. It is
self-contained (no internet, no separate image files — every chart is embedded),
opens in any browser, and is written for a **non-expert** reader: a verdict panel
up top with the recommended model and a one-line recommendation, then each stage's
findings with plain-English notes, then a glossary and a "run this on your own
data" how-to. That report *is* the project's face.

### The "edit one config" idea

Nothing in `src/` mentions "electricity". Every dataset-specific detail — where
the data is, what column to predict, what column gives time order, what to drop —
lives in the config file. Swap the config and the entire chain (both
classification *and* regression, automatically detected) runs on the new data,
saving a report named after that dataset. We ship five ready configs to prove it
(see `DATASETS.md`).

---

## Part 2 — The pipeline, stage by stage (layman → technical)

The pipeline is six stages, labelled **S0–S5** in the report. Data flows through
`run.py`, which threads a shared in-memory scratchpad (`ctx`) from stage to stage.

---

### Stage 0 — Data ingestion, profiling & the temporal split

**Layman.** Before any modelling we load the data, take its vital signs (rows,
features, missing values, class balance), figure out on our own whether this is a
*classification* problem (predict a category) or a *regression* problem (predict a
number), and — crucially — split train/test **by time**: the earliest 60% of rows
train the model, the latest 40% test it. That mimics reality, where you always
predict the future from the past, never the reverse.

**Why the time split matters (say this to the examiner).** A normal random split
lets the model peek at future rows while training — on time-ordered data that is a
subtle form of cheating that inflates the score. Sorting by time and cutting once
is what makes the later drift analysis honest.

**Technical (`src/data.py`, `src/config.py`).**

- `load_config` parses the YAML into typed dataclasses. `DatasetConfig` has fields
  `source`, `name`, `data_id`, `path`, `version`, `target`, `time_col`,
  `use_time_as_feature`, `drop_cols`. Each `dataset:` YAML key must match a field
  name exactly, because the loader does `DatasetConfig(**value)` — a typo raises a
  clear `TypeError` rather than silently ignoring the key.
- `load_raw` has three branches: OpenML by numeric `data_id` (most reliable),
  OpenML by `name`+`version`, or a local `csv` at `path`.
- `resolve_task` auto-detects the task from the target column: text / boolean /
  low-cardinality integer → classification; continuous numeric → regression. `task:
  auto` in the config uses this; you can override.
- `build_preprocessor` builds a leakage-safe `ColumnTransformer`: numeric columns
  get **median imputation + standard scaling**; categoricals get **most-frequent
  imputation + one-hot encoding**. A categorical with >50 distinct values triggers a
  warning (dense one-hot would explode memory — usually it's an ID you should drop).
- `prepare` sorts by `time_col` with a **stable mergesort**, drops the target, the
  time column, and any `drop_cols` from the feature matrix, `LabelEncoder`s the
  target for classification, and does the temporal split. If `time_col` is `null`
  it falls back to a random split and later **skips drift** gracefully.
- **Safety net:** a wrong `target`/`time_col`/`drop_cols` name stops immediately and
  prints the full list of available columns, so you copy the right name from the
  error. Result is a `DataBundle` carrying `X_train/X_test`, `y_train/y_test`,
  `preprocessor` (unfitted template), `task`, `has_time`, `numeric_cols`,
  `target_classes`, `time_test`, and a `profile` dict.

---

### Stage 1 — Regularized baselines (the honest yardstick)

**Layman.** Before the heavy machinery, we fit simple linear models with different
kinds of "restraint" (regularization) and record how they do. This gives an honest
floor: if a plain linear model gets 78%, the fancy models have to beat that to earn
their complexity. It also teaches what regularization *does*:

- **L2 (ridge-style):** keep every feature but shrink all the weights toward zero —
  gentle, smooth restraint.
- **L1 (lasso-style):** push some weights *exactly* to zero — it literally fires
  useless features (automatic feature selection).
- **ElasticNet:** a blend of both.

**The defensible substitution (examiner will ask).** The syllabus names
Ridge/Lasso/ElasticNet, which are *regression* method names. Our headline demo is
*classification*, so we use **Logistic Regression with penalty ∈ {L2, L1,
ElasticNet}** — that is the exact same regularization mathematics on the linear
term, just wrapped in a logistic loss instead of a squared loss. On the regression
configs (metro, appliances, bike) the pipeline uses the literal
`Ridge`/`Lasso`/`ElasticNet`. So we cover both.

**The "flat result is itself a finding" point.** On ELEC2 all four regularizers
score nearly identically at the default strength. That's not a bug — with abundant
rows and few features, the penalty barely moves the fit. To make the *mechanism*
visible anyway, we sweep the penalty strength (the **regularization path**) and
show L1 zeroing features and L2 shrinking norms as the strength changes.

**Technical (`src/baselines.py`).**

- `_classification_models` builds four `LogisticRegression`s at a common `C=1.0`:
  none / L2 / L1 / ElasticNet. `_regression_models` builds
  `LinearRegression`/`Ridge`/`Lasso`/`ElasticNet`.
- **scikit-learn 1.8 gotcha (important, we hit this live):** in sklearn ≥ 1.8,
  `LogisticRegression(penalty=...)` is deprecated and *silently overridden* by the
  new `l1_ratio` knob (default 0.0 = pure L2). Passing `penalty='l1'` therefore
  secretly ran L2, making all four models identical. `_sklearn_new_api()` detects
  the version and speaks the right dialect: on ≥1.8 we set `l1_ratio` (0=L2, 1=L1,
  0.5=EN) and `C=inf` for unregularized, using the `saga` solver whenever
  `l1_ratio>0`; on older sklearn we use the classic `penalty=`.
- `_coef_diagnostics` reports sparsity (fraction of zeroed coefficients), L1/L2
  norms, and the strongest features — the evidence that regularization *did*
  something.
- `_regularization_path` fits on an **internal 80/20 split of the training data
  only** (the real test set is never touched), sweeping `C` over `logspace(-3,2,12)`
  for L1 & L2 (classification) or `alpha` over `logspace(-4,1,12)` for Lasso & Ridge
  (regression), recording surviving-coefficient count, L2 norm, and a validation
  score at each strength.
- `_learning_curve` runs on the best baseline only (5 points × 3 folds,
  `TimeSeriesSplit` when there's time) — cheap bias/variance diagnostic. Every
  diagnostic is wrapped in try/except so it can never sink the run.
- Side effects: registers all results in `ctx['results']` and records
  `ctx['baseline_best']`.

---

### Stage 2 — Boosting ensembles + stacking (the workhorses)

**Layman.** Now the strong models. **Boosting** builds a team of small decision
trees where each new tree focuses on fixing the previous team's mistakes — three
industry-standard libraries do this: XGBoost, LightGBM, CatBoost. We add a
**RandomForest** as an older-style ensemble reference. Then **stacking**: train all
of them, and put a small "manager" model on top that learns how best to combine
their opinions — often a touch better than any single one.

**Technical (`src/boosting.py`).**

- Each booster is built by a lazily-imported builder (`_build_xgboost`, etc.). If a
  library isn't installed, that model is **skipped with a message** and the pipeline
  carries on — graceful degradation, not a crash. If *every* booster is missing, the
  stage falls back to the best Stage-1 baseline so downstream stages still have a
  model.
- Sensible fixed defaults (300 trees, lr 0.1, depth 6, subsample/colsample 0.9),
  `tree_method="hist"` for XGBoost to scale, `verbose=-1`/`verbose=0` to silence,
  and `allow_writing_files=False` for CatBoost so it doesn't litter a
  `catboost_info/` directory on Windows.
- **Stacking gotcha (great examiner answer):** sklearn's `StackingClassifier` builds
  meta-features with `cross_val_predict`, which requires the CV to be a *partition*
  (every row in exactly one validation fold). `TimeSeriesSplit` is **not** a
  partition (its earliest rows never get validated) and raises an error. So stacking
  uses **StratifiedKFold/KFold** internally. Time-honesty is preserved because the
  held-out *temporal* test set is never part of this CV — it's only scored at the
  very end. The meta-learner is deliberately a **regularized** linear model
  (LogisticRegression / RidgeCV) — the "RegStack" idea from Paper 1.
- **No oversubscription:** base learners use all cores (`n_jobs=-1`), stacking runs
  its folds sequentially (`n_jobs=1`), so we don't spawn cores² threads on a laptop.
- Feature-importance diagnostics captured for tree models.
- Side effects on `ctx`: `ensemble_best`, `best_booster`(+`_name`, **excludes
  stacking** because Stage 3 tunes a single booster), and `best_overall`/`best_model`
  (best across Stages 1–2, what Stages 4/5 will explain and monitor).

---

### Stage 3 — Hyperparameter-optimization comparison (the analytical centrepiece)

**Layman.** Every model has knobs (tree depth, learning rate…). "Tuning" means
searching for the best knob settings. There are three search styles:

- **Grid search** — try every combination on a coarse grid. Thorough but explodes
  combinatorially.
- **Random search** — throw darts at random settings. Surprisingly strong for the
  effort.
- **Bayesian optimization (Optuna)** — learn from the darts thrown so far to aim the
  next one smarter.

The whole project is built around one question: **"Is the smart, expensive method
actually worth it?"** So we don't just report the best score — we report the
**cost** (wall-clock seconds and number of model evaluations) next to it, and give
Random and Bayesian the **exact same budget** for a fair head-to-head.

**Technical (`src/tuning.py`).**

- Tunes `ctx['best_booster_name']` from Stage 2 (single best booster, stacking
  excluded). A search space is defined for every booster it knows (`_SEARCH_SPACES`),
  so whichever wins Stage 2 is tunable — the tool stays general.
- Each knob is declared **once** and consumed **three ways**: a discrete `grid` (for
  GridSearchCV), and continuous `low/high` (turned into scipy distributions for
  RandomizedSearchCV and into `trial.suggest_*` calls for Optuna). So all three
  explore the *same region of the same knobs* — the only thing that varies is the
  search brain.
- **One CV, one scorer, shared by all four strategies.** On time-ordered data that
  CV is `TimeSeriesSplit` — and unlike stacking, this is **safe**, because
  Grid/Random score folds with `cross_validate` (not `cross_val_predict`), so a
  non-partition splitter is fine. Scorer defaults to `roc_auc` (binary) /
  `roc_auc_ovr` (multiclass) / `neg_root_mean_squared_error` (regression); sklearn's
  "higher is better" convention means argmax always picks the winner, even for
  negated RMSE.
- **Honest scoring:** searches run with `refit=False`; we read the winning config
  from `cv_results_` ourselves, then **refit it once through the same `fit_evaluate`
  path as every other model** and score it on the untouched temporal test set — so
  its leaderboard row is directly comparable to Stages 1–2. The test set is never
  seen during search.
- **Equal-budget head-to-head:** Random and Bayesian each get exactly
  `budget_trials` (40) candidate evaluations. Grid is the exhaustive-coarse
  reference with its own (usually smaller) cost.
- **Cost fairness:** all Stage-3 fits use a fixed modest tree budget (`_TUNE_TREES =
  200`) and force the base learner single-threaded (`_SINGLE_THREAD`) so outer
  fold/candidate parallelism owns the cores — again avoiding cores² oversubscription.
- `_verdict` produces the plain-English takeaway: did tuning beat the default? did
  Bayesian beat random at equal budget? Promotes a tuned model to `ctx['best_overall']`
  if it now leads, so Stages 4/5 explain the genuinely strongest model.

---

### Stage 4 — SHAP explainability + a reliability layer

**Layman.** Two questions for a non-expert:

1. **"Why did the model decide that?"** SHAP produces an *itemized receipt*: for the
   model overall, which features matter most (global); and for one specific
   prediction, how each feature pushed the answer up or down (local).

2. **"When should we trust it, and when should a human look?"** We take the model's
   own confidence, flag the least-confident 20% of cases for human review, and then
   **prove the flag works** — that flagged pile really does contain a large share of
   all the mistakes, and the auto-accepted 80% is noticeably cleaner. That's the
   practical, deployable half of Paper 3.

**Technical (`src/explain.py`).**

- **Model-type-aware SHAP routing:** `TreeExplainer` if the model has
  `feature_importances_`, `LinearExplainer` if it has `coef_` (this mattered — a
  *linear* baseline won the first ELEC2 run!), else a model-agnostic `Explainer`
  fallback. If SHAP isn't installed, the stage degrades gracefully.
- SHAP is computed in the **transformed** feature space (the model lives behind the
  preprocessor) and labelled with the preprocessor's `get_feature_names_out()` so
  importances line up with real column names. `_shap_matrix` normalizes SHAP's many
  possible output shapes (list per class / `Explanation` object / 2-D / 3-D) into one
  `(n_samples, n_features)` matrix for the positive class. Bounded to ≤800 samples
  for speed.
- **Reliability layer needs no SHAP** — it uses the probabilities we already have.
  `_confidence` = max class probability. `_reliability` bins accuracy by confidence
  (quantile edges + `digitize`) to draw a calibration-style curve, then applies the
  flag rule: the least-confident `flag_frac` (0.20) go to review; it reports coverage,
  **error-capture rate**, and auto-accepted vs flagged accuracy. On the ELEC2 run,
  flagging ~20% caught ~34% of all errors.

---

### Stage 5 — Concept-drift monitoring + statistical significance (the differentiator)

**Layman.** A model trained once on the past — does it keep working as time moves
on? We cut the time-ordered test set into consecutive chunks ("this week, next
week…") and score a **frozen** (train-once) model on each in turn — a *walk-forward
/ prequential* evaluation. A downward trend means the model is ageing. In parallel
we run an **adaptive** copy that retrains as each chunk goes by, and race the two.
Separately, we measure how much the *inputs* themselves shifted using **PSI**
(Population Stability Index). And every "X beats Y" claim is backed by a **statistical
significance test**, not just a decimal.

**The key insight from the ELEC2 run (great to quote):** the inputs drifted *heavily*
(PSI rose well past the "large shift" threshold) yet accuracy *held up* — a nuanced,
real finding, not a canned "drift = bad" story.

**Technical (`src/drift.py`, `src/stats.py`).**

- **Skips cleanly** if there's no time column (drift is meaningless on a random split)
  — but still returns model-vs-model significance.
- **No look-ahead:** the adaptive model predicts chunk *i* strictly **before** it
  trains on chunk *i*. **Leakage-safe retrain:** each refit `clone()`s the *whole*
  pipeline, so the preprocessor's imputation/scaling statistics are relearned on the
  new window. Retrain modes: `none` / `periodic` (expanding window, keep all history)
  / `sliding` (fixed-length recent window).
- **PSI** per numeric feature vs the training window as reference (>0.1 moderate,
  >0.2 large shift); reports the mean-PSI trajectory and the worst feature/chunk.
- Optional **river ADWIN** change-point detector on the frozen error stream (graceful
  "unavailable" if river isn't installed).
- **Significance (`src/stats.py`):** **McNemar's test** for classification —
  explicitly named in the syllabus, a cheap high-credibility win. It looks only at
  the rows where two models *disagree*; if one is right on most of those, the gap is
  significant. Uses statsmodels (exact binomial when <25 disagreements, else
  continuity-corrected chi-square), with a scipy fallback. **Wilcoxon signed-rank**
  on paired absolute errors for regression. We run three head-to-heads: champion vs
  runner-up, champion vs the simple linear baseline ("is the complexity worth it?"),
  and frozen vs adaptive ("did retraining significantly help?").
- Never raises — every piece degrades to a partial-but-valid summary.

---

### The report (`src/report.py`) — the headline deliverable

**Layman.** Everything above is turned into one browser-openable HTML file with a
verdict on top and the details below, written so a non-expert can follow it.

**Technical.** `build(context, output_dir, title, cfg=None)` is a **pure-standard-
library** generator (only `base64`/`html`/`datetime`/`pathlib` — no jinja2, no
matplotlib at report time). It base64-embeds the PNG charts so the file is fully
self-contained, and even draws the Stage-1 regularization-path charts as **inline
SVG in pure Python** (baselines saves no PNGs). Design is a calm "diagnostic
instrument readout": verdict panel (recommended model + big primary score + status
pills + a recommendation box) → headline-finding callout → S0 dataset/config → S1
baselines → S2 ensembles → S3 tuning cost/benefit table → S4 SHAP + reliability
triage → S5 drift figures + per-chunk table + significance table → a "run this on
your own data" how-to → glossary. The report filename is slugified from the dataset
(`_dataset_slug`) so each dataset writes its own `results/report_<slug>.html`.

---

## Part 3 — Cross-cutting engineering (the details that impress)

These are the "software craft" points that separate this from a notebook. Mention
one or two unprompted.

- **One fit/evaluate path (`src/modeling.py`).** Every stage trains and scores
  through the *single* `fit_evaluate` function, so every row on the leaderboard is
  produced identically and is truly comparable. `ModelResult` carries both the
  in-memory fitted estimator (for later stages) and a JSON-safe `summary()` (for the
  report). Ranking (`pick_best`, `leaderboard`) is nan-safe and direction-aware via
  `primary_metric_name` + `higher_is_better`.

- **Leakage safety, three ways.** (1) The preprocessor is **cloned inside each
  model's Pipeline** so imputation/scaling stats are learned from training folds
  only — never from the whole dataset. (2) The **temporal split** means the model
  never sees future rows. (3) Tuning and the regularization path **never touch the
  test set** during search; winners are refit once and scored on the held-out set.

- **Graceful degradation everywhere.** Missing library → skip that model. SHAP
  absent → skip SHAP, keep reliability. No time column → skip drift, keep
  significance. river absent → skip ADWIN. No booster installed → fall back to best
  baseline. The pipeline *always* produces a valid report.

- **`ctx` vs `report_ctx` (a clean separation).** `ctx` is an in-memory scratchpad
  carrying fitted models and predictions between stages — never serialized.
  `report_ctx` holds only JSON-safe per-stage summaries — that's what gets saved to
  `results/tables/results.json` and handed to the report. The best model is
  additionally persisted to `results/best_model.pkl` (joblib) for reuse/deployment.

- **Two subtle CV facts** you can state confidently: stacking must use a *partition*
  CV (k-fold), not `TimeSeriesSplit`, because it relies on `cross_val_predict`;
  Grid/Random tuning *can* use `TimeSeriesSplit` because they use `cross_validate`.
  Knowing the difference is a strong signal you understand the tooling.

- **Reproducibility:** a single `random_state` (42) threads through every splitter,
  model, and sampler.

---

## Part 4 — The five references (all Q1) — the specific "point these out" ask

All five journals are **Q1** (top-quartile by SCImago Journal Rank). Lead with the
**venue + quartile**; that is the verified, high-credibility anchor. The DOIs below
are as recorded in your literature review — quote them, but the quartile is the part
that wins the point.

| # | Paper (short name) | Venue & year | Quartile (SJR) | Where it lives in our project |
|---|---|---|---|---|
| 1 | **RegStack** — regularized stacking | *Expert Systems with Applications* (ESWA), 2025 | **Q1** (SJR ≈ 1.85) | Stage 1 regularization as a first-class idea + Stage 2 stacking with a **regularized meta-learner** |
| 2 | **BO-XGBoost** — Bayesian-optimized XGBoost + SHAP | *Expert Systems with Applications* (ESWA), 2026 | **Q1** (SJR ≈ 1.85) | Stage 3 (Bayesian tuning of the boosting core vs grid) — our closest blueprint |
| 3 | **SHAP Error-Compensation** — SHAP → reliability | *IEEE Transactions on Artificial Intelligence* (IEEE TAI), 2025 | **Q1** (SJR ≈ 1.43; Q1 in AI and in CS-Applications) | Stage 4 reliability layer — flag unreliable predictions using the model's own signal |
| 4 | **Concept Drift, Continuous-Kernel** | *Information Sciences*, 2024 | **Q1** (SJR ≈ 1.80) | Stage 5 drift framing + the Electricity dataset itself |
| 5 | **Spectral-Clustering Tuning** | *IEEE Access*, 2025 | **Q1** (SJR ≈ 0.88) | Motivates the HPO discussion (literature context; the specific spectral method is not implemented) |

**DOIs (as in your review):** Paper 1 `10.1016/j.eswa.2025.127766` · Paper 2
`10.1016/j.eswa.2026.131260` · Paper 3 `10.1109/TAI.2024.3455313` · Paper 4
`10.1016/j.ins.2024.120649`. (Paper 5 — IEEE Access article page.)

**How to talk about each in one breath:**

- *RegStack (ESWA, Q1):* "Regularization shouldn't be an afterthought — it belongs
  both in the base learners and in the stacking meta-model. We took that directly:
  our stacker's meta-learner is a regularized linear model."
- *BO-XGBoost (ESWA, Q1):* "This is our closest blueprint — a boosting core tuned by
  Bayesian optimization and explained with SHAP. We extended it by *comparing*
  Bayesian against grid and random at equal budget and accounting for cost."
- *SHAP Error-Compensation (IEEE TAI, Q1):* "SHAP isn't just pretty plots — the signal
  can flag unreliable predictions. We implemented the *identify* half as a deployable
  human-in-the-loop triage rule and proved it captures a disproportionate share of
  errors."
- *Concept Drift (Information Sciences, Q1):* "This gave us the streaming/distribution-
  shift framing and the Electricity dataset. We implemented the practical
  monitor-and-retrain pattern rather than their continuous-kernel algorithm."
- *Spectral Tuning (IEEE Access, Q1):* "Part of the tuning literature we surveyed; it
  informs the HPO discussion but isn't in our code."

*Note on quartiles:* SJR values are the recent (≈2024) SCImago figures; treat them as
"comfortably Q1" rather than memorising exact decimals. If asked how you know they're
Q1: "SCImago Journal Rank divides each subject's journals into four quartiles by SJR;
these four sit in the top quarter of their categories."

---

## Part 5 — Syllabus & Course-Outcome mapping (CE29T Advanced ML)

Use this to show the project is not random — it deliberately spans the syllabus.

**Modules covered:**

- **Module 1 (Advanced Supervised / Ensembles):** Ridge/Lasso/ElasticNet (regression
  configs) and Logistic-Regression penalties (classification); XGBoost, LightGBM,
  CatBoost; **stacking**. ✔ heavily covered.
- **Module 2 (Model Evaluation & Selection):** Precision/Recall/F1/Fβ, ROC-AUC,
  PR-AUC; **McNemar's test**; **Grid / Random / Bayesian HPO**; learning curves. ✔
  heavily covered — this is the project's centre of gravity.
- **Module 5 (Explainability):** **SHAP** global + local. ✔ covered.
- **Module 6 (Drift & Monitoring):** **Concept drift**, prequential evaluation,
  retraining strategies, PSI, model monitoring. ✔ covered — our differentiator.
- **Modules 3 & 4 (Clustering; NLP/Topic-Modeling):** **not** covered — different data
  regime (unsupervised / text). Say so plainly if asked (see limitations).

**Lab practicals:** the project implements roughly **7 of the 10** listed practicals
(data prep/EDA, regularized regression, ensembles/boosting, evaluation metrics,
hyperparameter tuning, SHAP-based explainability, and a drift/monitoring practical).
The ones not implemented are the clustering/NLP-flavoured practicals.

**Course Outcomes:**

- **CO5 — directly satisfied:** statistical significance testing (McNemar/Wilcoxon) +
  hyperparameter optimization + concept-drift monitoring, all in one pipeline.
- **CO6 — directly satisfied:** an end-to-end, deployable system (saved model, a
  self-contained report, a reusable config-driven design) with an eye to
  ethics/sustainability (cost-accounted tuning = "don't burn compute for a marginal
  gain"; human-in-the-loop triage).

---

## Part 6 — Anticipated examiner questions (with crisp answers)

**Q: What's your actual contribution — this uses existing algorithms?**
A: The contribution is a *reusable system* and the *empirical study* it produces, not
a new algorithm. The novelty is integration: baselines → boosting → a cost-accounted
HPO comparison → SHAP + a deployable reliability rule → drift monitoring with
significance tests, all config-driven and dataset-agnostic, ending in one report a
non-expert can read.

**Q: Why Logistic Regression instead of Ridge/Lasso/ElasticNet?**
A: Same regularization mathematics (L1/L2/ElasticNet penalty on the linear term),
just a logistic loss because our headline demo is classification. On the regression
datasets the pipeline uses the literal Ridge/Lasso/ElasticNet. So both are covered.

**Q: Why a time-based split instead of a random split or k-fold?**
A: The data is time-ordered and the goal is to predict the future from the past. A
random split leaks future information into training and inflates the score; it also
makes drift analysis impossible. We sort by time and cut once.

**Q: All four regularized baselines scored almost the same — isn't that a failure?**
A: No — with many rows and few features the penalty barely moves the fit at the
default strength; that flatness is itself a finding. The regularization *path* (which
we plot) shows the mechanism clearly: L1 zeroing features, L2 shrinking the norm.

**Q: Is Bayesian optimization worth it?**
A: That's exactly the question we designed Stage 3 to answer, with cost measured
alongside score at equal budget. On ELEC2 the answer was "only a marginal gain over
random search for meaningfully more time" — which is a legitimate, useful result, not
a disappointment.

**Q: How do you avoid data leakage?**
A: Preprocessing is cloned inside each model's pipeline (stats learned per training
fold), the split is temporal, and the test set is never touched during tuning or the
regularization sweep — winners are refit once and scored on held-out data.

**Q: What is PSI and what threshold do you use?**
A: Population Stability Index — it compares each feature's distribution in a test
chunk against the training window. >0.1 is a moderate shift, >0.2 a large one. On
ELEC2 the inputs drifted well past 0.2 yet accuracy held — a nuanced result.

**Q: Why McNemar's test?**
A: It's the correct paired test for two classifiers on the same test set, and it's
named in the syllabus. It examines only the rows where the models disagree: if one is
right on most of those, the difference is statistically significant.

**Q: How does the reliability/triage layer work, and does it actually help?**
A: We rank predictions by the model's confidence, flag the least-confident 20% for
human review, and measure the error-capture rate. On ELEC2, ~20% flagged caught ~34%
of all errors, and the auto-accepted 80% was cleaner — so yes, it helps.

**Q: What happens if I give it a completely different dataset?**
A: Edit one config (source, target, optional time column, any `drop_cols`), run
`python run.py --config configs/yours.yaml`. It auto-detects classification vs
regression, swaps McNemar for Wilcoxon, drops the classification-only triage section,
and writes `results/report_<yourdataset>.html`. We ship five configs proving it.

**Q: Show me it's not overfit to one dataset.**
A: `DATASETS.md` + five configs span both tasks, categorical one-hot, and a leakage
demo (`bike_sharing` drops `casual`/`registered`/`instant`). The bike case is our
teaching example: left in, those columns give a fake ~1.0 R²; `drop_cols` makes it
honest.

---

## Part 7 — Honest limitations (preempt these; it reads as maturity)

- **Modules 3 & 4 (clustering, NLP/topic modelling) are out of scope** — different
  data regime. This is a supervised tabular pipeline by design.
- **Explainability is SHAP-only** — we did not add LIME, and did not use RFE / Mutual
  Information / Chi-square for feature selection (L1 provides embedded selection
  instead). Easy honest answer if pushed: "natural next additions."
- **The drift study implements a monitor-and-retrain pattern**, not the
  continuous-kernel algorithm of Paper 4 — practical, deployable, but not that
  paper's exact method.
- **Paper 5 (spectral tuning) is literature context**, not implemented.
- **Single demo dataset for the live numbers** (ELEC2); the other four configs are
  ready but need a one-file CSV download first (all free, no login).

---

## The two-minute opening you can memorise

"Our project is *TabML-Guard*, a reusable, config-driven pipeline for real-world
tabular prediction. You point it at any time-ordered dataset by editing one config,
and it runs the whole workflow — regularized baselines, boosting ensembles and
stacking, a Grid/Random/Bayesian tuning comparison that accounts for *cost* not just
accuracy, SHAP explainability with a human-in-the-loop reliability rule, and
concept-drift monitoring with McNemar significance tests — then produces one
self-contained HTML report with a plain-English verdict. Our demo is the ELEC2
electricity dataset. The headline findings were that a simple regularized model
stayed competitive, Bayesian tuning bought only a marginal gain over random search
at equal budget, and the inputs drifted heavily yet accuracy held. It's backed by
five Q1 papers, and it maps onto Modules 1, 2, 5 and 6 and satisfies CO5 and CO6."
