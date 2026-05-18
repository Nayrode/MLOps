# 03 — Train Model on Kubernetes

Trains a classifier on the preprocessed CS:GO match data using a Kubernetes Job, enabling distributed or GPU-accelerated training without tying up a local machine.

## Inputs

- `data/X_train.csv`
- `data/y_train.csv`

## Outputs

- Serialized model artifact (e.g. `model.pkl` or MLflow run)

## Run

```bash
kubectl apply -f job.yaml
kubectl logs -f job/<job-name>
```

## Notes

- Training script should load data from a shared volume or object storage (S3/GCS)
- Use a `ConfigMap` or environment variables for hyperparameters
