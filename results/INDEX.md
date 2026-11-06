# Results index

3 run(s) found in `results/`. Each row is one config; every dataset keeps its own folder with its report, figures, tables and saved model.

## All runs

| Dataset | Task | Rows | Features | Champion model | Metric | Score | Best HPO | Drift verdict |
|---|---|---:|---:|---|---|---:|---|---|
| [`appliances-energy`](appliances-energy/) | regression | 19,735 | 27 | `linear` | rmse | 94.1187 | bayes | inputs drifted, held |
| [`electricity`](electricity/) | classification | 45,312 | 7 | `logreg_l1` | roc_auc | 0.7884 | bayes | inputs drifted, held |
| [`metro-traffic`](metro-traffic/) | regression | 48,204 | 7 | `catboost_default` | rmse | 1893.9764 | default | inputs drifted, held |

## Per-run detail

### `appliances-energy`

- **Target / time column** — `Appliances` ordered by `date`
- **Champion** — `linear` at rmse = `94.1187` on the held-out period
- **Best tuning strategy** — bayes
- **Drift** — inputs drifted, held
- **Report** — [`report_appliances-energy.pdf`](appliances-energy/report_appliances-energy.pdf)
- **Artifacts** — `appliances-energy/figures/`, `appliances-energy/tables/`, `appliances-energy/best_model.pkl`

### `electricity`

- **Target / time column** — `class` ordered by `date`
- **Champion** — `logreg_l1` at roc_auc = `0.7884` on the held-out period
- **Best tuning strategy** — bayes
- **Drift** — inputs drifted, held
- **Report** — [`report_electricity.pdf`](electricity/report_electricity.pdf)
- **Artifacts** — `electricity/figures/`, `electricity/tables/`, `electricity/best_model.pkl`

### `metro-traffic`

- **Target / time column** — `traffic_volume` ordered by `date_time`
- **Champion** — `catboost_default` at rmse = `1893.9764` on the held-out period
- **Best tuning strategy** — default
- **Drift** — inputs drifted, held
- **Report** — [`report_metro-traffic.pdf`](metro-traffic/report_metro-traffic.pdf)
- **Artifacts** — `metro-traffic/figures/`, `metro-traffic/tables/`, `metro-traffic/best_model.pkl`

---

Regenerate with `python make_index.py`.
