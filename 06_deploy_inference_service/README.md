# 06 — Deploy Inference Service

Deploys the registered model as a real-time REST endpoint on Kubernetes using KServe.

## Inputs

- `inference_service.yaml` — KServe `InferenceService` manifest
- Model artefact accessible via the `storageUri` in the YAML (S3 or MLflow artifact store)

## What it does

1. Applies the `InferenceService` custom resource
2. Waits until the service is `Ready`
3. Prints the prediction endpoint URL (copy it into step 07's `.env`)

## Files

| File | Purpose |
|------|---------|
| `inference_service.yaml` | KServe manifest — update `storageUri` before applying |
| `deploy_inference_service.ipynb` | Apply, wait, print endpoint |

## Endpoint

```
POST /v1/models/csgo-match-predictor:predict
Content-Type: application/json

{"instances": [[elo_diff, winrate_10_diff, winrate_30_diff, experience_diff, rank_diff, h2h_winrate, map_Cache, ...]]}
```

## Run

Update `storageUri` in `inference_service.yaml`, then:

```bash
uv sync
uv run jupyter lab
```

Open `deploy_inference_service.ipynb` and run all cells.
