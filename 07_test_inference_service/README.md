# 07 — Test Inference Service

Smoke-tests the deployed inference service with 10 sample rows from the test set.

## Inputs

- Live endpoint from step 06
- `data/X_test.csv` / `data/y_test.csv`

## What it does

1. Samples 10 rows from `X_test`
2. POSTs them to the inference endpoint
3. Asserts: response has 10 predictions, all values are `0` or `1`
4. Reports how many match expected labels

## Environment

```
INFERENCE_URL=http://<kserve-ingress>/v1/models/csgo-match-predictor:predict
```

Defaults to `http://localhost:8080/...` if unset (useful with port-forward).

## Run

```bash
cp .env.example .env
uv sync && uv run python test_inference_service.py
```
