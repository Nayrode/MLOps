# 05 — Upload Model

Registers the trained model artifact in a model registry (MLflow) and tags it with the run metadata.

## Inputs

- Trained model artifact from step 03
- Evaluation metrics from step 04

## What it does

- Logs model to MLflow Model Registry
- Attaches metrics, parameters, and dataset metadata to the run
- Transitions model to `Staging` stage if evaluation thresholds are met

## Run

```bash
uv sync
uv run jupyter lab
```

Open `upload_model.ipynb` and run all cells.

## Notes

- Requires `MLFLOW_TRACKING_URI` set in environment
