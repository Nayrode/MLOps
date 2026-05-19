# 03 — Train Model on Kubernetes

Trains a `LogisticRegression` classifier on the preprocessed CS:GO match data. Supports two modes: local (fallback, runs in the notebook directly) and Kubernetes (builds a Docker image, submits a Job, streams logs).

## Model

`sklearn.linear_model.LogisticRegression` — v1 baseline.  
Hyperparameters are configurable via environment variables / ConfigMap. If `data/best_params.json` exists (from step 08), those values take priority.

## Inputs

- `data/X_train.csv`
- `data/y_train.csv`
- `data/best_params.json` _(optional — from step 08)_

## Outputs

- `data/model.pkl` — serialised model (local mode)
- `/model/model.pkl` — serialised model (Kubernetes PVC)
- MLflow run with `train_accuracy` metric

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script (container entrypoint) |
| `Dockerfile` | Builds the training image |
| `job.yaml` | Kubernetes Job + ConfigMap |
| `train_model_kubernetes.ipynb` | Orchestration: local training → build → push → submit → monitor |

## Run

```bash
uv sync
uv run jupyter lab
```

Open `train_model_kubernetes.ipynb`. Run the **local training cell** for a quick baseline, or the **Kubernetes cells** to submit the Job.

## Notes

- Update `REGISTRY` in the notebook and `image:` in `job.yaml` before pushing
- The Kubernetes Job reads data from `csgo-data-pvc` and writes the model to `csgo-model-pvc`
- Set `MLFLOW_TRACKING_URI` to point at your MLflow server
