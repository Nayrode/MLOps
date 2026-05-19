# 04 — Evaluate Model

Evaluates the trained model on the held-out test set, logs metrics to MLflow, and saves them to disk for step 05.

## Inputs

- `data/model.pkl` — full Pipeline from step 03
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

```
MLFLOW_TRACKING_URI=http://localhost:5000   # optional, defaults to local mlruns/
```

## Run

```bash
cp .env.example .env
uv sync && uv run python evaluate_model.py
```
