# 00 — DVC Pull

Pulls the versioned dataset from a DVC remote. Data is currently git-tracked so this step is optional — `data/results.csv` is already present in the repository.

## Inputs

- DVC remote configured in `.dvc/config`
- `DVC_S3_ACCESS_KEY` / `DVC_S3_SECRET_KEY` in `.env` (only needed if using an S3 remote)

## Outputs

- `data/results.csv` — raw CS:GO match results

## Run

```bash
# If data is git-tracked (default):
git pull   # results.csv comes with the repo

# If using a DVC remote:
cp .env.example .env   # fill in credentials
uv sync && uv run python dvc_pull.py
```

## Dependencies

- `dvc` — pipeline orchestration and optional remote storage
- `python-dotenv` — loads `.env` into environment variables
