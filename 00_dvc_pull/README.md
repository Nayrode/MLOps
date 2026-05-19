# 00 — DVC Pull

Pulls the versioned dataset from a remote GitHub repo + S3 backend using `dvc.api.DVCFileSystem`. No local DVC config needed — credentials are passed at runtime via a `.env` file.

## Inputs

- GitHub repo `Nayrode/MLOps.git` at git ref `cap4`
- `DVC_S3_ACCESS_KEY` and `DVC_S3_SECRET_KEY` in `.env`

## Outputs

- `data/results.csv` — raw CS:GO match results

## Run

```bash
cp .env.example .env   # fill in credentials
uv sync && uv run python dvc_pull.py
```

## Environment

```
DVC_S3_ACCESS_KEY=...
DVC_S3_SECRET_KEY=...
```

## Dependencies

- `dvc[s3]` — DVC with S3 remote support
- `python-dotenv` — loads `.env` into environment variables
- `pandas` — reads and saves the CSV
