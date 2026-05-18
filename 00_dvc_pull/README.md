# 00 — DVC Pull

Pulls the versioned dataset from remote storage (S3) into `data/` using DVC.

## Inputs

- `.dvc` tracking files committed in the repo
- DVC remote configured in `.dvc/config` (S3)

## Outputs

- `data/results.csv` — raw CS:GO match results

## Run

```bash
uv sync
uv run jupyter lab
```

Open `dvc_pull.ipynb` and run all cells.

## Dependencies

- `dvc[s3]` — DVC with S3 remote support
