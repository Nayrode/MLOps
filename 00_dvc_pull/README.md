# 00 — DVC Pull

Pulls the versioned dataset from a remote GitHub repo + S3 backend using `dvc.api.DVCFileSystem`. No local DVC config needed — credentials are passed at runtime via a `.env` file.

## Inputs

- GitHub repo `USERNAME/CS-yr-match-predictor` at git ref `cap4`
- `DVC_S3_ACCESS_KEY` and `DVC_S3_SECRET_KEY` in `.env`

## Outputs

- `data/raw/results.csv` — raw CS:GO match results

## Run

```bash
uv sync
uv run jupyter lab
```

Open `dvc_pull.ipynb` and run all cells.

## Environment

Create a `.env` file in this directory (never commit it):

```
DVC_S3_ACCESS_KEY=...
DVC_S3_SECRET_KEY=...
```

## Dependencies

- `dvc[s3]` — DVC with S3 remote support
- `python-dotenv` — loads `.env` into environment variables
- `pandas` — reads and saves the CSV
