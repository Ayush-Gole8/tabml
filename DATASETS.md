# Running TabML-Guard on Other Datasets

TabML-Guard is dataset-agnostic. Nothing in `tabml_guard/` knows about "electricity"
specifically — every dataset detail comes from the config file or CLI arguments. To point the
whole pipeline (baselines → ensembles → HPO → SHAP + reliability → drift +
significance → HTML & PDF reports) at a new dataset, you edit a config and run:

```bash
tabml-guard run --config configs/<dataset>.yaml
# or: python run.py --config configs/<dataset>.yaml
```

The reports are written into isolated per-dataset directories:

- `config.yaml` → `results/electricity/report_electricity.html` (and `.pdf`)
- `configs/occupancy_detection.yaml` → `results/occupancy/report_occupancy.html`
- `configs/metro_traffic.yaml` → `results/metro-traffic/report_metro-traffic.html`
- `configs/appliances_energy.yaml` → `results/appliances-energy/report_appliances-energy.html`
- `configs/bike_sharing.yaml` → `results/bike-sharing/report_bike-sharing.html`

This folder ships with five ready-to-run configs. One (electricity) runs
straight from OpenML with no download; the other four each need a single CSV
placed in `data/` first (all are free, no login).

---

## The datasets

| Config | Task | Time column | Target | Rows | Source | Why it's here |
|---|---|---|---|---|---|---|
| `config.yaml` (electricity) | Classification | `date` | `class` (UP/DOWN) | 45,312 | OpenML `electricity` (id 151) | The baseline case from the report. Runs with no download. |
| `configs/occupancy_detection.yaml` | Classification | `date` | `Occupancy` (0/1) | ~20,560 | UCI *Occupancy Detection* (357) | A second classification story on a clean, all-numeric signal — full McNemar + drift path. |
| `configs/metro_traffic.yaml` | **Regression** | `date_time` | `traffic_volume` | 48,204 | UCI *Metro Interstate Traffic Volume* (492) | Flips the pipeline to RMSE/MAE/R² + Wilcoxon. Has a few low-cardinality text columns (one-hot handles them). |
| `configs/appliances_energy.yaml` | **Regression** | `date` | `Appliances` (Wh) | 19,735 | UCI *Appliances Energy Prediction* (374) | Dense, fully numeric — best case for the PSI drift trajectory and the regularization path. |
| `configs/bike_sharing.yaml` | **Regression** | `dteday` | `cnt` | 17,379 | UCI *Bike Sharing* (275), hourly `hour.csv` | The `drop_cols` demo: removes an ID and two leakage columns so the score stays honest. |

Together these exercise every branch of the tool: both task types, the
significance test swap (McNemar for classification, Wilcoxon for regression),
the classification-only confidence-triage layer, categorical one-hot encoding,
and leakage removal via `drop_cols`.

---

## How to get each CSV

**Occupancy Detection** — UCI dataset 357. It ships as three comma-separated
files with headers (`datatraining.txt`, `datatest.txt`, `datatest2.txt`).
Concatenate them in that order into `data/occupancy.csv`, keeping one header
row. Columns: `date, Temperature, Humidity, Light, CO2, HumidityRatio,
Occupancy`.

**Metro Interstate Traffic Volume** — UCI dataset 492. Save the single file as
`data/metro_traffic.csv`. Columns: `holiday, temp, rain_1h, snow_1h,
clouds_all, weather_main, weather_description, date_time, traffic_volume`.

**Appliances Energy Prediction** — UCI dataset 374. Save the file
`energydata_complete.csv` as `data/appliances_energy.csv`. Target is
`Appliances`; columns `rv1` and `rv2` are deliberate random noise you can leave
in or drop.

**Bike Sharing** — UCI dataset 275. Use the **hourly** file `hour.csv`, saved
as `data/bike_sharing.csv`. Columns: `instant, dteday, season, yr, mnth, hr,
holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual,
registered, cnt`.

Each config's header comment repeats its download note and column list, so you
don't need this file open while setting up.

---

## What each run should show (the insight story)

**electricity (classification).** The regularized logistic baseline stays
competitive with the tuned boosters — a genuine "simple model is enough here"
finding — Bayesian tuning buys only a marginal gain over random search at equal
budget, inputs drift heavily (PSI rises) yet accuracy holds, and the confidence
triage flags the least-certain slice for review.

**occupancy (classification).** A much cleaner signal (light and CO₂ are very
predictive of a room being occupied), so expect high AUC across the board and a
smaller model spread. The interesting angles become SHAP (which sensor drives
the call) and whether accuracy stays flat across the day/night drift chunks.

**metro_traffic (regression).** The report switches to RMSE / MAE / R² with no
config change beyond the dataset. The significance section now runs a Wilcoxon
signed-rank test on per-row errors instead of McNemar, and the
classification-only confidence-triage section is dropped automatically. Traffic
has strong daily/weekly cycles, so watch the drift chunks.

**appliances_energy (regression).** Every feature is numeric, so the PSI drift
trajectory and the regularization path are at their most informative here. A
good case for discussing which sensors matter (SHAP) and whether the `rv1`/`rv2`
noise columns correctly rank near-zero.

**bike_sharing (regression).** The teaching case for leakage. `casual` +
`registered` sum exactly to `cnt`, and `instant` is a row counter — left in, the
model scores a meaningless near-perfect R². With
`drop_cols: [instant, casual, registered]` the evaluation is honest and the
seasonal/weather features do the real work.

---

## Config options you'll actually touch

Every key under `dataset:` maps to a field the loader understands. The ones
that matter when adding a dataset:

- **`source`** — `openml` or `csv`.
- **`name`** / **`data_id`** — for OpenML. `data_id` is the numeric id (e.g.
  `151` for electricity) and is the most reliable identifier; if both are set,
  `data_id` wins. `name` uses `version` (default 1).
- **`path`** — the CSV path when `source: csv` (relative to the project root).
- **`target`** — the column to predict. Task is auto-detected from it: text /
  boolean / few-valued integer columns → classification; continuous numeric →
  regression. Set `task:` explicitly to override.
- **`time_col`** — the column that gives chronological order. When set, rows are
  sorted by it and the split is temporal (earliest rows train, latest rows
  test), which is what makes the drift analysis meaningful. **Leave it `null`
  and the pipeline falls back to a random split and skips drift** — a quick way
  to see the graceful-degradation path on any dataset.
- **`use_time_as_feature`** — normally `false` (the timestamp is used only for
  ordering, not as a model input).
- **`drop_cols`** — columns to exclude: IDs, free text, and above all **leakage**
  (anything that encodes the answer). This is what keeps `bike_sharing` honest.

The other sections (`split`, `models`, `tuning`, `drift`, `report`,
`random_state`, `output_dir`) can stay at the shipped defaults; they behave the
same across datasets.

---

## Two built-in safety nets

**Wrong column name?** If `target`, `time_col`, or a `drop_cols` entry doesn't
exist in the file, the loader stops immediately and prints the full list of
available columns, so you can copy the correct name straight from the error —
no guessing. (A missing `drop_cols` name is a warning, not a fatal error.)

**High-cardinality categorical?** Any text column with more than 50 distinct
values triggers a warning, because dense one-hot encoding would add that many
columns and can blow up memory. If you see it, either drop that column (it's
usually an ID) via `drop_cols`, or group its rare levels before running. The
five shipped datasets are all safe on this front.

---

## Quick start (a regression example)

```bash
# 1. Put the file in place
#    -> data/metro_traffic.csv   (from UCI dataset 492)

# 2. Run the pipeline on it
tabml-guard run --config configs/metro_traffic.yaml

# 3. Open the report
#    -> results/metro-traffic/report_metro-traffic.html
#    -> results/metro-traffic/report_metro-traffic.pdf
```

That's the whole workflow. To try your own data, copy the closest config,
change `path`/`target`/`time_col` (and `drop_cols` if there's any leakage), and
run.
