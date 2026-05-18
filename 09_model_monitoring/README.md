# 09 — Model Monitoring

Monitors the deployed model for data drift and performance degradation over time.

## What it monitors

- **Data drift**: distribution shift in incoming feature values vs. training baseline (e.g. Evidently or Alibi Detect)
- **Prediction drift**: shift in the model's output distribution
- **Performance**: rolling accuracy / AUC on labeled production data when ground truth becomes available

## Inputs

- Production request logs from the inference service (step 06)
- Training baseline statistics from `data/X_train.csv`
- Ground truth labels (delayed, once match results are known)

## Outputs

- Drift report (HTML or JSON)
- Alerts if drift exceeds configured thresholds
- Metrics pushed to Grafana / Prometheus

## Run

```bash
uv sync
uv run jupyter lab
```

Open `model_monitoring.ipynb` and run all cells.
