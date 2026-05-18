# 07 — Test Inference Service

Smoke-tests the deployed inference service with sample match data to verify the endpoint is healthy and returning valid predictions.

## Inputs

- Deployed inference service URL (step 06)
- Sample rows from `data/X_test.csv`

## What it does

- Sends prediction requests to the live endpoint
- Asserts response shape and value range (probabilities in [0, 1])
- Compares a small batch of predictions against expected labels to catch silent regressions

## Run

```bash
uv sync
uv run jupyter lab
```

Open `test_inference_service.ipynb` and run all cells.

## Notes

- Set `INFERENCE_URL` in environment before running
