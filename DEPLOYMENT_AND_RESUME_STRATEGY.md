# TabML-Guard: Deployment, Public Showcasing & Resume Strategy

**Author:** Ayush Gole  
**Project:** TabML-Guard — Configuration-Driven, Drift-Aware & Explainable Tabular AutoML Pipeline  
**Target Roles:** Machine Learning Engineer (MLE), MLOps Engineer, Applied Data Scientist, AI Systems Engineer  

---

## 1. Scope, Motive & Vision Analysis

### 1.1 The Motive: Escaping the "Notebook ML" Trap
The vast majority of candidate machine learning resumes list projects like *"Customer Churn Prediction with XGBoost"* or *"House Price Prediction using Random Forest"*. Interviewers and engineering managers see thousands of these. Almost all of them share fundamental flaws:
1. **Silent Data Leakage & Shuffling Bias:** They randomly shuffle time-ordered data, allowing future information to leak into the past.
2. **Reflexive Over-Engineering:** They jump directly to complex boosting models or spend hours tuning hyperparameters without evaluating if a regularized linear model (`Ridge`/`Lasso`/`LogisticRegression`) performs identically or better.
3. **No Cost Accounting:** Hyperparameter Optimization (HPO) leaderboards report accuracy in isolation, hiding whether a 0.2% improvement cost 20 minutes of CPU/GPU compute.
4. **Blind Accuracy vs. Triage:** A 78% accuracy means 22% of predictions fail silently in production. Standard projects do not know *which* predictions to distrust.
5. **Post-Deployment Amnesia:** Models are treated as static objects. When input distributions drift or concepts shift over time, they silently degrade.

### 1.2 The Vision & Core Thesis: A Universal, Data-Agnostic Engine
**TabML-Guard is fundamentally data-agnostic.** The 5 datasets analyzed in our repository are **validation benchmarks and empirical stress tests**, not the boundaries of the system. The codebase in `src/` contains zero hardcoded assumptions about any domain, schema, or dataset.

Whether the input is:
- High-frequency financial transactions or crypto price logs,
- Server infrastructure / DevOps metrics and IoT telemetry streams,
- E-commerce customer churn, conversion events, or clickstream tables,
- Hospital patient vitals, telemetry, or healthcare readmission sequences,
- Or classical non-temporal tabular datasets (where temporal drift gracefully disables),

**TabML-Guard operates as a universal plug-and-play diagnostic framework** that enforces honest, reproducible, and defensible ML workflows by construction:
- **One File Interface (`config.yaml`) / Single Function Call:** Point the engine at any CSV or OpenML identifier, define the `target` and optional `time_col`, and the pipeline handles profiling, task detection, preprocessing, model exploration, tuning, explainability, and drift analysis automatically.
- **Leakage-Safe Architecture:** Preprocessors are fit strictly on training splits and cloned per cross-validation fold; temporal splits preserve chronological causality across any arbitrary time horizon.
- **Cost-Accounted HPO (Stage 3):** Compares Default, Grid, Random, and Bayesian (Optuna TPE) search at an *equal evaluation budget*, explicitly plotting Score vs. Wall-Clock Compute.
- **Reliability & Human-in-the-Loop Triage (Stage 4):** Employs SHAP and model confidence calibration to flag the bottom 20% uncertain predictions, empirically proving that this slice captures ~34% of real-world errors.
- **Prequential Drift & Statistical Significance (Stage 5):** Tracks Population Stability Index (PSI) and races frozen vs. sliding-window retrained models across time chunks, backed by McNemar's / Wilcoxon signed-rank significance tests.
- **Academic Pedigree:** Backed by an IEEE Access research paper synthesizing five Q1 peer-reviewed papers (RegStack, BO-XGBoost, SHAP Error-Compensation, Concept Drift Kernel Learning, Spectral HPO).

---

## 2. The "Frontend" Question in a Data-Agnostic Context: What Do We Actually Need?

Because **TabML-Guard is 100% data-agnostic**, anyone should be able to run it on their own dataset. That fundamentally reshapes the frontend and deployment question:

> **Do we need a frontend?**  
> If "frontend" means building a full-stack React/Node.js web application from scratch, **no — that is an engineering distraction.**  
> However, if "frontend" means giving recruiters, researchers, and engineers **an effortless way to test TabML-Guard on ANY dataset of their choice**, then having an interactive web presence is a huge asset — provided it is architected correctly.

### 2.1 Why Full-Stack React Web Apps Fail for Generic AutoML
1. **The Latency & Timeout Dilemma:** Training 5 models (XGBoost, LightGBM, CatBoost, RandomForest, Stacking) plus 40 Bayesian Optuna trials on a 50,000-row table takes 2–10 minutes. Free/cheap cloud hosting (Render, Vercel, Hugging Face CPU) enforces hard timeouts (30s to 120s) and 512MB–2GB RAM limits. If a user uploads a 100MB CSV on a React app, the background worker will crash or timeout.
2. **Data Privacy & Enterprise Reality:** Real data scientists and ML engineers never upload proprietary financial, healthcare, or telemetry data to a third-party hobby web app. They use **CLIs and Python packages** (`pip install ...`) running inside their own secure environments.
3. **Resume Dilution:** An ML engineer resume evaluated on React CSS buttons gives zero signal regarding MLOps, statistical validation, or ML system design.

---

### 2.2 The Three Interaction Modes for a Generic, Data-Agnostic Tool

To satisfy every persona (recruiters with 30 seconds, data scientists with their own data, and production ML engineers), we structure TabML-Guard into **three complementary interfaces**:

```
                                  ┌───────────────────────────────────────────────┐
                                  │          TabML-Guard Engine (src/)            │
                                  │   (Universal, Config-Driven, Data-Agnostic)   │
                                  └───────────────────────────────────────────────┘
                                         │                    │                 │
                ┌────────────────────────┘                    │                 └────────────────────────┐
                ▼                                             ▼                                          ▼
   [Persona 1: The Recruiter]                     [Persona 2: The Practitioner]              [Persona 3: The Production Eng]
┌─────────────────────────────────┐           ┌──────────────────────────────────┐       ┌────────────────────────────────┐
│   Interactive Web Studio (HF)   │           │      Python Package & CLI        │       │  Data-Agnostic MLOps Service   │
│  (Hugging Face Space / Streamlit)│          │     (`pip install tabml-guard`)   │       │   (FastAPI + Docker Container) │
├─────────────────────────────────┤           ├──────────────────────────────────┤       ├────────────────────────────────┤
│ • Upload ANY user CSV           │           │ • Run on 10M rows locally        │       │ • POST /api/audit (upload data)│
│ • Dropdowns: target & time_col  │           │ • 100% data privacy / on-premise │       │ • Real-time confidence triage  │
│ • "Quick Audit" (~30s runtime)  │           │ • CI/CD pipeline integration     │       │ • Streaming PSI drift detector │
│ • 1-click Benchmark Presets     │           │ • Zero cloud upload limits       │       │ • Auto-generated Swagger UI    │
│ • Live HTML report & best model │           │ • Scriptable: `tabml-guard run`  │       │ • Ready for Kubernetes / Cloud │
└─────────────────────────────────┘           └──────────────────────────────────┘       └────────────────────────────────┘
```

1. **Interface 1: The "Bring-Your-Own-Data" (BYOD) Web Studio (Streamlit / Gradio on Hugging Face Spaces)**
   - **For recruiters & live evaluation:** They don't need to install Python or clone git.
   - **How it works with ANY dataset:**
     - The user drags and drops **any CSV** (or selects an OpenML dataset).
     - The app parses the schema and presents dynamic dropdowns:
       - **Target Column:** Auto-detects possible targets; user selects one.
       - **Time Column:** Optional dropdown (select timestamp or choose "No temporal order").
       - **Compute Budget:** *"Fast Audit (30s)"* (Ridge/Lasso, LightGBM, 5 Optuna trials) or *"Deep Audit (2 min)"*.
     - **Fallback Presets:** For users browsing on mobile or without a CSV handy, a one-click *"Try a Benchmark"* dropdown loads the pre-configured Electricity, Metro Traffic, or Appliances Energy datasets instantly.
     - **The Deliverable:** Directly renders the interactive HTML report inside the browser and offers one-click downloads for `best_model.pkl` and `report.pdf`.

2. **Interface 2: The Open-Source Python Package & CLI (`pip install tabml-guard`)**
   - **For data scientists & ML engineers:** Like `evidently`, `great_expectations`, `ydata-profiling`, or `cleanlab`.
   - Allows users to run TabML-Guard on massive datasets locally or in corporate VPCs without data ever leaving their machine:
     ```bash
     pip install tabml-guard
     tabml-guard run --data my_data.csv --target churn --time timestamp
     ```
     Or as a Python module:
     ```python
     from tabml_guard import TabMLGuard

     guard = TabMLGuard.from_df(df, target="response", time_col="created_at")
     results = guard.fit()
     guard.save_report("audit.html")
     ```
   - **Resume Signal:** Positions you as a **Developer Tool Builder / Framework Engineer**, not just someone running ad-hoc scripts.

3. **Interface 3: The Data-Agnostic MLOps Service (FastAPI + Docker)**
   - Exposes TabML-Guard as an asynchronous microservice:
     - `POST /audit`: Ingests any CSV and training configuration, returns a `job_id`.
     - `GET /audit/{job_id}/status`: Polls progress across S0–S5 stages.
     - `GET /audit/{job_id}/report`: Returns the generated interactive HTML report.
     - `POST /models/{job_id}/predict`: Serves real-time inference with calibrated confidence and human-in-the-loop triage routing.
     - `POST /models/{job_id}/drift`: Calculates live PSI drift on incoming batches against the training baseline.

---

## 3. Deployment Options & Comparative Matrix

| Deployment Route | Primary Purpose | Tech Stack | Cost | Hosting Platform | Resume Signal |
|---|---|---|---|---|---|
| **1. GitHub Pages Showcase** | Interactive Report Hub & Benchmark Gallery | HTML5, CSS3, SVG, GitHub Pages | $0 (Free) | GitHub Pages (`username.github.io/tabml-guard`) | **High:** Demonstrates professional product delivery, documentation, and reporting. |
| **2. Production Microservice** | Real-Time Serving, Confidence Triage, Drift API | Python, FastAPI, Pydantic, Docker | $0 (Free tier) | Hugging Face Spaces (Docker) or Render | **Highest:** Proves production MLOps, API design, serialization, and containerization. |
| **3. Hugging Face Space** | Interactive Model Inspector & What-If Playground | Streamlit / Gradio | $0 (Free) | Hugging Face Spaces | **High:** Live interactive demo recruiters can test without technical setup. |
| **4. PyPI Package / CLI** | Reusable Python Library & Command-Line Utility | `pip`, `setuptools`, `argparse` | $0 (Free) | PyPI (`pip install tabml-guard`) | **Very High:** Proves open-source engineering maturity and code modularity. |

---

## 4. Architectural Blueprints for Deployment

### 4.1 Blueprint 1: GitHub Pages Report Showcase (Static Web Hub)
Every run in TabML-Guard produces an interactive HTML report. We can expose an index landing page that links to all benchmark reports:

```
public_html/ (deployed to gh-pages branch)
├── index.html                  <-- Portfolio Landing Page with comparison matrix & links
├── electricity/
│   ├── report.html             <-- Interactive HTML Report (Classification)
│   └── report.pdf              <-- Downloadable Formal PDF Report
├── occupancy/
│   └── report.html             <-- Environmental Sensor Benchmark
├── metro-traffic/
│   └── report.html             <-- Regression Benchmark (Traffic Volume)
├── appliances-energy/
│   └── report.html             <-- High-dimensional Regression Benchmark
└── bike-sharing/
    └── report.html             <-- Temporal Leakage Prevention Benchmark
```

#### GitHub Actions Workflow (`.github/workflows/deploy-pages.yml`):
```yaml
name: Deploy Reports to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: setup-python@v5
        with:
          python-version: '3.11'

      - name: Generate Reports & Site
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          python make_site.py --output-dir public_html

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: 'public_html'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

---

### 4.2 Blueprint 2: Production Model Serving Microservice (FastAPI + Docker)
Deploying `best_model.pkl` behind a production FastAPI microservice demonstrates true MLOps capabilities.

#### Implementation: `api.py`
```python
"""TabML-Guard Inference & Guardrail Microservice.

Provides low-latency inference, confidence-based triage routing,
and Population Stability Index (PSI) drift monitoring for deployed models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="TabML-Guard Serving & Monitoring API",
    description="Production model serving with confidence-based human-in-the-loop triage and drift monitoring.",
    version="1.0.0",
)

# Load default model and run metadata
MODEL_DIR = Path("results/electricity")
PIPELINE = joblib.load(MODEL_DIR / "best_model.pkl")
RESULTS_CTX = json.loads((MODEL_DIR / "tables" / "results.json").read_text(encoding="utf-8"))

CONFIDENCE_THRESHOLD = 0.65  # Bottom quantile routed to human review


class InferenceRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        example=[
            {
                "day": "2",
                "period": 0.0212,
                "nswprice": 0.0514,
                "nswdemand": 0.4255,
                "vicprice": 0.0034,
                "vicdemand": 0.4229,
                "transfer": 0.4149,
            }
        ],
    )


class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float]
    triage_status: str  # "AUTO_ACCEPTED" or "FLAGGED_FOR_HUMAN_REVIEW"
    triage_reason: Optional[str]


class DriftCheckRequest(BaseModel):
    features: List[Dict[str, float]]


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "champion_model": RESULTS_CTX.get("baselines", {}).get("best", "unknown"),
        "task": RESULTS_CTX.get("task", "classification"),
    }


@app.post("/predict", response_model=List[PredictionResponse], tags=["Inference"])
def predict(payload: InferenceRequest):
    """Run inference with confidence scoring and automated triage routing."""
    try:
        df = pd.DataFrame(payload.data)
        preds = PIPELINE.predict(df)
        probas = PIPELINE.predict_proba(df) if hasattr(PIPELINE, "predict_proba") else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}")

    results = []
    for i in range(len(df)):
        pred = preds[i]
        conf = float(np.max(probas[i])) if probas is not None else None

        # Guardrail: Flag low-confidence predictions for human triage
        if conf is not None and conf < CONFIDENCE_THRESHOLD:
            status = "FLAGGED_FOR_HUMAN_REVIEW"
            reason = f"Model confidence ({conf:.2f}) falls below safety threshold ({CONFIDENCE_THRESHOLD:.2f})."
        else:
            status = "AUTO_ACCEPTED"
            reason = None

        results.append(
            PredictionResponse(
                prediction=int(pred) if isinstance(pred, (np.integer, int)) else str(pred),
                confidence=round(conf, 4) if conf is not None else None,
                triage_status=status,
                triage_reason=reason,
            )
        )
    return results


@app.get("/report", response_class=HTMLResponse, tags=["Audit"])
def get_report():
    """Serve the self-contained HTML audit report."""
    report_file = MODEL_DIR / "report_class.html"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Audit report not generated.")
    return HTMLResponse(content=report_file.read_text(encoding="utf-8"))
```

#### Production Containerization: `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn pydantic

# Copy project files
COPY src/ ./src/
COPY results/ ./results/
COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 4.3 Blueprint 3: The "Bring-Your-Own-Data" Web Studio (Streamlit on Hugging Face Spaces)

To showcase universal data-agnosticism on the web without crashing on free cloud tiers, build an interactive Streamlit app hosted on Hugging Face Spaces with **two dynamic execution modes**:

#### Feature 1: The "Upload Any Dataset" Engine (BYOD)
- The user uploads **any CSV file** or enters an **OpenML Dataset ID/URL**.
- The app immediately parses headers and renders schema dropdowns:
  - **Target Column:** user picks the target label (or auto-detected).
  - **Time Column:** user picks the chronological column (or clicks *"No time ordering"* to run standard cross-validation).
  - **Drop Columns:** multiselect to remove IDs or known leakage.
  - **Audit Depth:**
    - **"Fast Audit (~30s)"**: Runs S0 (Data) $\to$ S1 (Regularized Baselines) $\to$ S2 (LightGBM + XGBoost) $\to$ S3 (5 Optuna trials) $\to$ S4 (SHAP Summary) $\to$ S5 (Prequential Drift). Runs live within browser HTTP limits!
    - **"Deep MLOps Audit (~2 min)"**: Full 5-booster ensemble + Stacking + 40 Optuna trials.
- **Output:** Live progress bar through stages S0–S5 $\to$ embeds the generated HTML report directly in an `iframe` $\to$ provides download buttons for `report.html`, `report.pdf`, and `best_model.pkl`.

#### Feature 2: Instant Benchmark Presets
- For recruiters or hiring managers who don't have a dataset on hand, provide a **"Try a Preloaded Benchmark"** dropdown:
  - *Electricity* (Classification, 45k rows, OpenML)
  - *Metro Traffic* (Regression, 48k rows)
  - *Appliances Energy* (High-dimensional regression & drift)
- Clicking a preset loads the full pre-computed diagnostic report, interactive SHAP waterfall plots, and drift curves instantly (<1s).

---

### 4.4 Blueprint 4: Python Package & CLI Distribution (`pip install tabml-guard`)

For enterprise MLOps and applied ML roles, releasing a Python package and CLI proves you can build production-grade developer tooling:

```bash
# 1. Install directly from GitHub or PyPI
pip install tabml-guard

# 2. Run on any dataset from the terminal
tabml-guard run --data my_timeseries.csv --target churn --time timestamp

# 3. Or use inside Python scripts / Jupyter notebooks
```

```python
from tabml_guard import TabMLGuard

# Data-agnostic execution on any Pandas DataFrame
guard = TabMLGuard(
    target="default_flag",
    time_col="transaction_date",
    task="auto"
)
results = guard.fit(df)

# Export deliverables
guard.save_report("credit_risk_audit.html")
model = guard.get_best_model() # scikit-learn Pipeline ready for production
```

---

## 5. Resume Positioning & Portfolio Playbook

### 5.1 Project Title on Resume
Choose a title that conveys systems engineering and enterprise rigor:
- **TabML-Guard: Industrial AutoML Engine with Cost-Accounted HPO & Drift Guardrails**  
  *or*  
- **TabML-Guard: Configuration-Driven Tabular ML Framework with Explainable Triage & Drift Monitoring**

### 5.2 Resume Bullet Points (STAR / XYZ Format)

#### For Machine Learning Engineer (MLE) Roles:
> - **Architected TabML-Guard**, a configuration-driven, leakage-safe AutoML pipeline supporting classification and regression across time-ordered tabular datasets.
> - **Benchmarked Bayesian Optimization (Optuna TPE) vs. Grid/Random search** under strictly matched evaluation budgets; proved Bayesian HPO yielded marginal gains (<0.2% ROC-AUC) on low-signal tables despite 15x compute overhead, championing cost-aware model selection.
> - **Engineered an explainable reliability & triage layer** using calibrated probabilities and SHAP attributions, successfully capturing **34% of real-world model errors** by escalating only the bottom 20% uncertain predictions to human review.
> - **Built a low-latency model serving microservice (FastAPI, Docker)** with automated schema validation, confidence-based triage routing, and real-time Population Stability Index (PSI) drift monitoring.

#### For MLOps / Production ML Roles:
> - **Developed an end-to-end reproducible ML pipeline** enforcing temporal splitting, in-fold preprocessing, and statistical validation (McNemar’s and Wilcoxon tests) to prevent silent data leakage.
> - **Implemented prequential walk-forward drift monitoring** comparing frozen vs. sliding-window retraining strategies, detecting covariate shift via multi-chunk Population Stability Index (PSI).
> - **Containerized inference pipeline via Docker** and automated reporting workflows generating standalone, self-contained HTML/PDF diagnostic dashboards published via GitHub Actions CI/CD.
> - **Authored technical research paper** submitted to IEEE Access evaluating algorithmic performance, compute-cost efficiency, and distribution shift on real-world energy and sensor streams.

---

## 6. Examiner & Interview Defense Q&A Bank

### Q1: "Why did you build your own pipeline instead of just using AutoGluon, TPOT, or auto-sklearn?"
> **Answer:** *"AutoML tools like AutoGluon or auto-sklearn are designed to answer one question: 'What pipeline maximizes the metric under a compute limit?' They do not answer the three questions that determine production viability: (1) Did that tuning pay for itself relative to a simple regularized baseline? (2) Which specific predictions are untrustworthy and should be routed to a human? and (3) Will this model survive concept drift next month? TabML-Guard was built as an audit and guardrail framework that treats the model's cost, reliability, and temporal stability as primary deliverables alongside raw accuracy."*

### Q2: "Why did regularized linear models beat tuned gradient boosting on two of your three benchmark datasets?"
> **Answer:** *"This is one of the headline findings of our study. Gradient boosting excels when there are complex non-linear feature interactions and high signal-to-noise ratios. On smaller or noisy tables (such as Appliances Energy), boosting ensembles overfit to local temporal noise. A heavily regularized linear model (L1/ElasticNet) provided smoother inductive bias, achieving lower RMSE on the temporally held-out test stream. It highlights why starting with regularized baselines is an engineering necessity rather than an academic formality."*

### Q3: "Why did you split by time instead of using standard k-fold cross-validation?"
> **Answer:** *"Standard k-fold cross-validation shuffles data randomly across time. In time-ordered tabular data (financial, electricity, sensory), shuffling leaks future correlation patterns into training folds, producing artificially optimistic scores that collapse in production. Furthermore, shuffling completely masks concept drift. By strictly enforcing temporal splits (earliest rows train, latest rows test), our evaluation mirrors genuine production deployment and enables prequential drift simulation."*

### Q4: "How does your reliability triage layer actually work in practice?"
> **Answer:** *"In production, autonomous models fail silently on ambiguous edge cases. Our reliability layer buckets predictions by calibrated model confidence and SHAP attributions. In our experiments on the electricity dataset, the bottom 20% least-confident predictions had an error rate of ~50% (equivalent to random guessing), whereas the top 80% auto-accepted predictions maintained over 80% accuracy. By isolating and routing that bottom 20% to human review, we captured 34% of all errors while maintaining autonomous processing for the vast majority of traffic."*

### Q5: "What is the difference between covariate drift and concept drift in your pipeline?"
> **Answer:** *"Covariate shift (feature drift) is a change in the input distribution P(X), which we measure using the Population Stability Index (PSI) across time chunks. Concept drift is a change in the mapping P(Y|X), which we monitor via prequential (test-then-train) performance decay. Crucially, our empirical results showed that high PSI (>0.25) did not automatically cause accuracy decay, demonstrating that input distribution shifts do not always degrade model utility if the decision boundary remains invariant."*

---

## 7. Phased Implementation Roadmap

```
Phase 1: Zero-Cost Showcase (Today)
├── Sync run.py to output both HTML and PDF reports
├── Build make_site.py to generate public_html/ with an executive landing page
└── Deploy to GitHub Pages (Live URL for Resume)

Phase 2: Production Microservice & Docker (Day 2)
├── Write api.py (FastAPI endpoints: /predict, /predict/triage, /monitor/drift)
├── Write Dockerfile and test local container build
└── Deploy free container on Hugging Face Spaces or Render

Phase 3: GitHub & Portfolio Polish (Day 3)
├── Update README.md with live demo links, architecture diagram, and badges
├── Add GitHub Actions CI workflow (ruff linting, test suite, auto-deployment)
└── Update LinkedIn & Resume with STAR-formatted bullet points
```

