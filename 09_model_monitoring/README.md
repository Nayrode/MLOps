# 09 — Model Monitoring

Monitors the deployed model for data drift and performance degradation using Evidently.

## What it monitors

| Check | Tool | Trigger |
|-------|------|---------|
| Feature distribution drift | Evidently `DataDriftPreset` | >20% of features drifted |
| Data quality | Evidently `DataQualityPreset` | Missing values, type mismatches |
| Rolling accuracy / AUC | sklearn metrics | Manual inspection |

## Inputs

- `data/X_train.csv` — training baseline (reference distribution)
- `data/X_test.csv` — current / production data (swap with live logs in production)
- `data/y_test.csv` — ground truth labels
- `data/model.pkl` — model from step 03

## Outputs

- `data/reports/drift_report.html` — full Evidently HTML report
- Console alert if drift share exceeds `DRIFT_THRESHOLD` (default 0.20)
- Accuracy and ROC AUC printed for the current data window

## Run

```bash
uv sync
uv run jupyter lab
```

Open `model_monitoring.ipynb` and run all cells.

## Notes

- In production, replace `X_test.csv` with a rolling window of logged inference requests
- Ground truth labels become available once match results are confirmed
