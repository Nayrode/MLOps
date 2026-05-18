# 04 — Evaluate Model

Evaluates the trained model on the held-out test set and logs metrics.

## Inputs

- `data/X_test.csv`
- `data/y_test.csv`
- Trained model artifact from step 03

## Metrics

- Accuracy
- ROC AUC
- Precision / Recall / F1
- Confusion matrix

## Outputs

- Metrics logged to MLflow (or printed to stdout)
- Optional: evaluation report saved to `data/evaluation.json`

## Run

```bash
uv sync
uv run jupyter lab
```

Open `evaluate_model.ipynb` and run all cells.
