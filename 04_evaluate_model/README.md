# 04 — Evaluate Model

Evaluates the trained model on the held-out test set, logs metrics to MLflow, and saves them to disk for step 05.

## Inputs

- `data/model.pkl` — trained model from step 03
- `data/X_test.csv`
- `data/y_test.csv`

## Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Share of correctly predicted match outcomes |
| ROC AUC | Area under the ROC curve |
| Precision / Recall / F1 | Per-class breakdown |

## Outputs

- `data/evaluation.json` — `{"accuracy": ..., "roc_auc": ...}`
- `data/confusion_matrix.png` — confusion matrix plot
- MLflow run with metrics + confusion matrix artifact

## Environment

Create `.env` (optional — defaults to local `mlruns/`):

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Run

```bash
uv sync
uv run jupyter lab
```

Open `evaluate_model.ipynb` and run all cells.
