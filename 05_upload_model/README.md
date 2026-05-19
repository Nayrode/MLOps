# 05 — Upload Model

Registers the trained model in the MLflow Model Registry and promotes it to `Staging` if it clears the accuracy threshold.

## Inputs

- `data/model.pkl` — full Pipeline from step 03
- `data/evaluation.json` — metrics from step 04

## What it does

1. Loads model + metrics
2. Logs Pipeline to MLflow Model Registry as `csgo-match-predictor`
3. If `accuracy >= 0.60`, transitions the new version to `Staging`

## Outputs

- New registered model version in the MLflow registry
- Version stage: `Staging` (if threshold met) or `None`

## Environment

```
MLFLOW_TRACKING_URI=http://localhost:5000   # optional, defaults to local mlruns/
```

## Run

```bash
cp .env.example .env
uv sync && uv run python upload_model.py
```
