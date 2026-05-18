# 06 — Deploy Inference Service

Deploys the registered model as an online inference service on Kubernetes using KServe (or Seldon Core).

## Inputs

- Model URI from MLflow Registry (step 05)

## What it does

- Creates or updates an `InferenceService` custom resource
- Exposes a REST endpoint for real-time match outcome predictions
- Handles model download from the registry on startup

## Run

```bash
kubectl apply -f inference_service.yaml
kubectl get inferenceservice
```

## Endpoint

```
POST /v1/models/<model-name>:predict
Content-Type: application/json

{"instances": [[elo_diff, winrate_10_diff, ..., map_Vertigo]]}
```
