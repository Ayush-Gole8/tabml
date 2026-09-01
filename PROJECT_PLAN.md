# TabML-Guard — Design & Implementation Plan

**An Explainable, Drift-Aware AutoML Pipeline for Real-World Time-Ordered Tabular Data**

> Working name for the tool: **TabML-Guard** (feel free to rename). The word "Guard"
> captures the two things that make this project stand out from a normal model
> comparison: it *guards* against unreliable predictions (explainability + reliability
> flagging) and against silent decay after deployment (drift monitoring).

Course: Advanced Machine Learning, Semester VII (CMPN)
Assignment title submitted: *Explainable and Drift-Aware Hyperparameter Optimization of
Ensemble Machine Learning Models for Real-World Tabular Prediction.*

---

## 0. Read this first — what kind of project this is

This is an **implementation project**, not a paper survey and not a one-off experiment.

The deliverable is a **reusable software tool**: a configuration-driven pipeline that
accepts *any* time-ordered tabular dataset, automatically detects whether the task is
classification or regression, runs the full advanced-ML chain end-to-end, and emits a
single self-contained **HTML report** for the end user.

One dataset (electricity price movement, **ELEC2**) is the *headline demonstration*.
A second dataset of a different task type is the *generality proof*. Neither dataset
*is* the project — the pipeline is the project.

**The sentence you say to evaluators:**

> "We built a reusable, explainable, drift-aware machine-learning pipeline for tabular
> prediction. It automates regularized baselines, gradient-boosting ensembles, a
> three-way hyperparameter-optimization comparison, SHAP interpretability with a
> reliability layer, and concept-drift monitoring — and it generates a comparative
> report automatically. We demonstrate it on electricity price-movement prediction and
> validate its generality on a second dataset. Our contribution is the integrated
> system and the empirical findings it produces about the
> accuracy–cost–interpretability–robustness trade-offs."

We are **not** claiming a new algorithm. The contribution is the *integration* and the
*empirical study the tool produces*. That is exactly the right scope for a 3–4 week
undergraduate mini-project, and it is defensible.

---

## 1. What the tool does (functional requirements)

A user (a data analyst, another student, a small business) should be able to:

1. Point the tool at a dataset (a CSV file **or** a named OpenML dataset) by editing a
   short config — no code changes.
2. Tell it the **target column** and, optionally, a **time column** (for temporal
   ordering / drift). If no time column exists, the tool falls back to a stratified
   split and simply reports that drift analysis is not applicable.
3. Get the tool to **auto-detect the task**: classification if the target is
   discrete/categorical, regression if it is continuous. Metrics and models adapt
   automatically.
4. Run one command and receive:
   - a console log of progress,
   - saved artifacts (plots, metrics tables, the trained best model),
   - a single **HTML report** that tells the whole story.

Non-goals (say these out loud so scope stays honest): no deep learning, no real
streaming infrastructure (we *simulate* the stream from the time order), no claim of
state-of-the-art accuracy, no hyperparameter search over dozens of models.

---

## 2. The pipeline, stage by stage — techniques + what we borrow from each paper

Each stage below lists: **what it does**, **the concrete technique/library**, and the
**paper it draws from** (with DOI). This mapping is what makes the literature review and
the implementation one continuous story instead of two disconnected deliverables.

### Stage 0 — Data ingestion, EDA, temporal split
- **What:** load data; profile it (shape, dtypes, missingness, target distribution,
  class balance, time span); auto-detect task type; clean (impute, encode categoricals,
  scale for linear models); split **by time** (earliest X% train, latest Y% test) rather
  than shuffling, so the evaluation mimics deployment.
- **Technique:** pandas for profiling; `sklearn.pipeline` + `ColumnTransformer` for a
  leakage-safe preprocessing pipeline (fit on train only); a `task_detector` heuristic
  (numeric + many unique values → regression; else classification).
- **Why temporal split matters:** it is the honest setup for a time-ordered dataset and
  it is what later lets us *see* drift. Random shuffling would hide drift entirely.

### Stage 1 — Regularized baselines (bias–variance foundation)
- **What:** establish the honest yardstick with simple, regularized linear models before
  reaching for heavy machinery. Show coefficient **shrinkage** (L2) and **sparsity**
  (L1), and plot **learning curves** to visualize the bias–variance trade-off.
- **Technique (classification):** `LogisticRegression` with `penalty in {l1, l2,
  elasticnet}`. **Technique (regression):** `Ridge`, `Lasso`, `ElasticNet`.
  Regularization strength chosen by cross-validation (`LogisticRegressionCV` /
  `RidgeCV`, etc.).
- **Borrowed from — Paper 1, RegStack (ESWA 2025, DOI 10.1016/j.eswa.2025.127766):**
  the idea of *combining L1 and L2 regularization* and using regularized models inside a
  larger ensemble. We adopt their regularization emphasis for the baseline layer and
  their **stacking** idea in Stage 2.
- **Defense note:** Ridge/Lasso/ElasticNet are *regression* estimators; their honest
  classification analog is Logistic Regression with the same penalties — identical
  regularization mathematics, applied to a classifier. Be ready to say this.

### Stage 2 — Ensemble / boosting models (+ a stacking meta-model)
- **What:** the workhorses. Train modern gradient-boosting models and an older ensemble
  baseline, plus a stacking meta-model that combines them (the RegStack idea).
- **Technique:** `XGBoost`, `LightGBM`, `CatBoost`, `RandomForest` (older-ensemble
  baseline), and a `StackingClassifier`/`StackingRegressor` whose **meta-learner is a
  regularized linear model** (ties Stage 1 and Stage 2 together — this is literally
  "RegStack" in spirit).
- **Metrics — classification:** Accuracy, Precision, Recall, F1, F-beta, ROC-AUC,
  PR-AUC, confusion matrix. **Regression:** MAE, RMSE, R². Everything computed on the
  temporally held-out test set.
- **Borrowed from — Paper 1 (stacking) and Paper 2, BO-XGBoost (ESWA 2026, DOI
  10.1016/j.eswa.2026.131260):** XGBoost as the core learner and the general boosting
  approach.

### Stage 3 — Hyperparameter-optimization comparison (the analytical centerpiece)
- **What:** take the strongest booster from Stage 2 and tune it **four ways**, recording
  **both** predictive score **and** computational cost (wall-clock seconds + number of
  model evaluations). Answer a real question, not a trivial one.
- **Technique:** Default params → `GridSearchCV` → `RandomizedSearchCV` → **Bayesian
  optimization via `Optuna`** (TPE sampler). All use the *same* time-aware CV splitter
  (`TimeSeriesSplit`) and the *same* search space where comparable, so the comparison is
  fair.
- **The question we answer:** *"Does Bayesian optimization justify its cost?"* — i.e.
  performance gained per unit of tuning compute. We report a **cost-vs-accuracy plot**
  and a one-line verdict.
- **Borrowed from — Paper 2, BO-XGBoost (ESWA 2026):** they use Bayesian optimization to
  tune XGBoost and validate with nested/5-fold CV, comparing against grid search and
  other baselines. We reproduce that *comparison methodology* (not their copper-smelting
  application) and add explicit cost accounting.

### Stage 4 — Explainability (SHAP) + a reliability layer
- **What:** open the black box. Global feature importance, a few local explanations, and
  — the ambitious bit — use model confidence / SHAP signal to **flag predictions likely
  to be wrong**, then verify that flagged predictions really do have higher error.
- **Technique:** `shap.TreeExplainer` on the best boosting model → beeswarm + bar
  (global), waterfall/force (local). **Reliability analysis:** bucket test predictions by
  model confidence (or by a SHAP-derived anomaly score), compute error rate per bucket,
  and show error rises in the low-confidence bucket (a *reliability/calibration* curve).
- **Borrowed from — Paper 3, SHAP Error Compensation (IEEE TAI 2025, DOI
  10.1109/TAI.2024.3455313):** their key idea is that SHAP is not just for pretty plots —
  it can *identify and compensate unreliable predictions*. We implement the "identify"
  half: use explanation/confidence signals to isolate the predictions you should not
  trust. This is what turns "we made SHAP plots" into a real objective.

### Stage 5 — Concept-drift monitoring (the differentiator)
- **What:** simulate deployment over time. Train on the early period; then walk forward
  through later data in chunks, measuring performance in each chunk; detect when
  performance degrades (concept drift) and when input distributions shift (feature
  drift); finally show that a **periodically retrained** model beats a **frozen** one.
- **Technique:**
  - *Prequential (test-then-train) evaluation* over time-ordered chunks.
  - *Performance drift detectors:* ADWIN / DDM / Page-Hinkley (via the `river` library),
    or a simple rolling-accuracy monitor if `river` is unavailable.
  - *Feature/covariate drift:* Population Stability Index (PSI) and KS-test comparing
    each later window against the training window.
  - *Response comparison:* Static model vs. sliding-window-retrained model vs. periodic
    full retrain — plotted as accuracy-over-time.
- **Borrowed from — Paper 4, Concept Drift with Continuous Kernel Learning (Information
  Sciences 2024, DOI 10.1016/j.ins.2024.120649):** their problem framing (streaming data,
  distribution changes, ensemble adaptation) and — conveniently — their use of the
  **Electricity** dataset. We do not reproduce their continuous-kernel algorithm; we
  implement the practical *monitor-and-retrain* pattern their problem motivates.

### Cross-cutting — Statistical significance
- **What:** don't just say model A beat model B — test whether the difference is real.
- **Technique:** **McNemar's test** (`statsmodels`) on the paired predictions of two
  classifiers on the same test set; for regression, a paired test on per-sample errors.
- **Why it's in scope:** your syllabus explicitly lists McNemar's test. Using it is a
  cheap, high-credibility win — and it only works for classification, which is one more
  reason the ELEC2 classification framing is the right call.

### Paper 5 — where it lives
- **Paper 5, Spectral Clustering Parameter Tuning (IEEE Access 2025, DOI
  10.1109/ACCESS.2025.3623882):** stays primarily in the **literature review**. Its
  transferable lesson — *parameter choice, not just algorithm choice, drives performance*
  — is exactly the thesis of Stage 3. Optional nod in code: an unsupervised EDA step that
  clusters the feature space, but this is not required and should not bloat the build.

---

## 3. Architecture & repository layout

Everything is modular so the dataset is swappable in one config edit. Each stage is a
function taking prepared data and returning a results object the report can render.

```
tabml-guard/
├── config.yaml            # the ONLY thing a user edits: dataset, target, time col, budgets
├── requirements.txt
├── run.py                 # entry point: reads config, runs stages, writes report
├── src/
│   ├── config.py          # load/validate config; sensible defaults
│   ├── data.py            # load (CSV/OpenML), profile, auto-detect task, preprocess, temporal split
│   ├── baselines.py       # Stage 1: regularized linear models + learning curves
│   ├── boosting.py        # Stage 2: XGB/LGBM/CatBoost/RF + stacking meta-model
│   ├── tuning.py          # Stage 3: Default/Grid/Random/Optuna + cost accounting
│   ├── explain.py         # Stage 4: SHAP global/local + reliability buckets
│   ├── drift.py           # Stage 5: prequential eval, detectors, PSI, static-vs-retrained
│   ├── stats.py           # McNemar's + paired tests
│   ├── metrics.py         # task-aware metric computation (classification vs regression)
│   ├── plots.py           # all figures saved to results/figures/
│   └── report.py          # assembles the self-contained HTML report
├── results/
│   ├── figures/           # PNGs
│   ├── tables/            # CSVs
│   ├── best_model.pkl
│   └── report.html        # THE deliverable
├── notebook.ipynb         # a narrated walkthrough for the live demo
└── README.md              # setup, run instructions, results summary
```

**Design rules that keep it generalisable:**
- No dataset name appears anywhere except `config.yaml`.
- Every stage function signature is task-agnostic; task type flows through as a flag.
- Preprocessing is fit on training data only (no leakage), inside an sklearn `Pipeline`.
- Randomness is seeded everywhere for reproducibility.

**`config.yaml` sketch:**
```yaml
dataset:
  source: openml          # openml | csv
  name: electricity       # or path: data/mydata.csv
  target: class
  time_col: date          # null if none -> drift analysis skipped
task: auto                # auto | classification | regression
split:
  train_frac: 0.6
  test_frac: 0.4          # temporal, no shuffle
tuning:
  budget_trials: 40       # Optuna trials; grid/random matched by count/time
models:
  boosters: [xgboost, lightgbm, catboost, random_forest]
  stacking: true
drift:
  n_chunks: 10
  retrain: sliding        # none | periodic | sliding
report:
  title: "TabML-Guard Report"
```

---

## 4. The end-user final report (the headline deliverable)

The report is a **single self-contained HTML file** — open in any browser, no server,
nothing to install. It is written *for a non-expert end user* first, with technical
depth available as you scroll. This is what you put on screen in the evaluation.

**Report sections, top to bottom:**

1. **Executive summary (auto-written from results).** 4–6 plain-English sentences: what
   task, best model, its headline metric, whether tuning was worth it, whether the model
   is reliable, and whether it drifts over time. A non-expert should understand the
   verdict here without reading further.
2. **Dataset overview.** Rows/columns, task type (auto-detected), time span, target
   distribution / class balance, missingness table, a note on the temporal split.
3. **Model leaderboard.** Sortable table: every model with all task-appropriate metrics,
   best row highlighted. This is the "what beat what" at a glance.
4. **Best-model deep dive.** For classification: ROC curve, PR curve, confusion matrix,
   threshold note. For regression: predicted-vs-actual, residual plot.
5. **Bias–variance panel.** Learning curves for the regularized baselines; a sentence on
   over/underfitting and what regularization did to the coefficients (sparsity plot).
6. **Hyperparameter-optimization verdict.** The cost-vs-accuracy chart (score on one
   axis, tuning seconds / #evaluations on the other) + a one-line verdict: *"Bayesian
   optimization reached X in N trials / T seconds vs grid search's Y in M / T′ — worth
   it / not worth it because …"*
7. **Explainability.** Global SHAP beeswarm + bar; 2–3 local explanations for
   representative cases (a confident-correct, a confident-wrong, a borderline).
8. **Reliability.** Error-rate-by-confidence-bucket chart: shows the model *knows when it
   doesn't know* (or doesn't), and how many predictions you'd flag for human review.
9. **Drift over time.** Accuracy-over-chunks line chart with detector-flagged drift
   points; PSI table for top features; static-vs-retrained comparison line.
10. **Statistical significance.** McNemar's verdict for the top-2 models: is the winner
    *significantly* better, or is the gap noise?
11. **Reproducibility footer.** Config used, library versions, random seed, run
    timestamp. Signals rigor.

**Why this format is impactful:** it makes the tool feel like a product ("feed it data →
get a decision-ready report"), it front-loads a plain-language verdict for the examiner,
and every claim is backed by a figure. It is far more memorable than scrolling a
notebook.

*(If you want a PDF too, the HTML converts cleanly — we can add that at the end.)*

---

## 5. Datasets

**Primary (headline demo) — ELEC2 / "electricity" (OpenML).**
~45,312 rows, 8 features, binary target (price UP vs DOWN vs a moving average),
half-hourly records → real time order. Chosen because it is the exact dataset in the
concept-drift paper, it is classification (enables McNemar's), it is time-ordered
(enables drift), and it loads in one line.

**Secondary (generality proof) — a regression, time-ordered set.**
Recommended: **Bike Sharing Demand** (hourly counts, clean temporal order) — this proves
the tool auto-switches to **regression**, exercising the whole task-agnostic design.
Alternative if you'd rather stay in classification+drift: the **Airlines** delay dataset
(a classic drift benchmark). One run on the secondary set, one report, one slide: "same
tool, different domain and task type, still works."

---

## 6. Two-day build schedule

**Day 1 — the spine must run end-to-end (even if rough).**
- Environment + `config.yaml` + `data.py` (load, profile, auto-detect, temporal split). ~1h
- Stage 1 baselines + learning curves. ~1h
- Stage 2 boosters + stacking + full metrics leaderboard. ~1.5h
- Stage 3 tuning comparison with cost accounting. ~2h
- McNemar's test wired in. ~0.5h
- A first ugly HTML report that renders stages 1–3. ~1h
- *End of Day 1: a complete, defensible project exists.*

**Day 2 — depth, generality, polish.**
- Stage 4 SHAP + reliability. ~2h
- Stage 5 drift + static-vs-retrained. ~2h
- Report polish (executive summary, styling, all figures). ~1.5h
- Secondary-dataset generality run. ~1h
- README + notebook walkthrough + rehearse the 4-line pitch. ~1.5h

**Triage rule if time runs short:** the spine (Stages 1–3 + report + McNemar's) is
non-negotiable and already sufficient. Stage 5 drift is the highest-value *extra*; Stage
4 reliability layer is next; the secondary-dataset run is the easiest thing to shorten
(one run, one screenshot).

---

## 7. How to run (on your machine)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py --config config.yaml                    # writes results/report.html
```

`requirements.txt`: `scikit-learn xgboost lightgbm catboost optuna shap statsmodels
river pandas numpy matplotlib seaborn pyyaml jinja2 joblib`.

---

## 8. Anticipated examiner questions (and your answers)

- *"How is this an implementation and not a study?"* → It's a reusable tool; the study is
  its *output*. Show the config swap and the second-dataset report.
- *"Why Logistic Regression when the title says Ridge/Lasso?"* → Same L1/L2 regularization
  math; Ridge/Lasso are the regression form, Logistic-with-penalty is the classification
  form. The tool uses whichever the task calls for.
- *"Is your accuracy state-of-the-art?"* → Not the goal. The goal is the integrated
  trade-off analysis: accuracy vs tuning cost vs interpretability vs robustness to drift.
- *"How do you know drift is real and not noise?"* → Detectors (ADWIN/PSI) flag it, and
  the retrained-vs-static gap quantifies its cost.
- *"Did you just run models once?"* → Time-aware CV in tuning, McNemar's for
  significance, seeded runs for reproducibility.

---

## 9. Honest limitations (state these — it builds credibility)

- Drift is *simulated* from historical time order, not a live production stream.
- SHAP reliability flagging is a practical heuristic, not the full error-compensation
  algorithm of the IEEE TAI paper.
- Two datasets prove generality in principle, not universally.
- No deep-learning models; scope is classical/ensemble ML per the syllabus.
