# 03 — Train Model on Kubernetes

Trains a full sklearn `Pipeline` (StandardScaler + OneHotEncoder + classifier) on the preprocessed CS:GO match data. Supports two modes: local and Kubernetes Job.

## Model

Default: `XGBClassifier` — test accuracy 80.4%, ROC AUC 0.8915.
Alternative: `LogisticRegression` — test accuracy 76.9%, ROC AUC 0.851.

Switch via `MODEL_TYPE` env var (`xgboost` or `logreg`).
If `data/best_params.json` exists (from step 08) and its `model_type` matches, those params take priority.

## Inputs

- `data/X_train.csv`
- `data/y_train.csv`
- `data/best_params.json` _(optional — from step 08, must match `model_type`)_

## Outputs

- `data/model.pkl` — full serialised Pipeline (preprocessor + classifier)
- MLflow run with `train_accuracy` metric (if `MLFLOW_TRACKING_URI` is set)

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script (container entrypoint) |
| `Dockerfile` | Builds the training image |
| `job.yaml` | Kubernetes Job manifest |

## Run

**Local:**

```bash
uv sync
DATA_DIR=$(pwd)/../data MODEL_DIR=$(pwd)/../data MODEL_TYPE=xgboost uv run python train.py
```

**Kubernetes:**

```bash
docker build -t YOUR_REGISTRY/csgo-trainer:latest .
docker push YOUR_REGISTRY/csgo-trainer:latest
kubectl apply -f job.yaml
kubectl logs -f job/csgo-train-job
```

## Notes

- Update `image:` in `job.yaml` with your registry before pushing
- The Job reads from `csgo-data-pvc` and writes to `csgo-model-pvc`
