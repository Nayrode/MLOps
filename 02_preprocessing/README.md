# 02 — Preprocessing

Encodes categorical features and produces a chronological train/test split ready for model training.

## What it does

- **Map encoding**: one-hot encodes `_map` (10 CS:GO maps) — low cardinality, map can affect match outcome
- **Team names**: skipped — 1,200+ unique values, already captured by Elo/winrate features, would break on unseen teams at inference
- **NaN handling**: none needed; dataset is fully clean (asserted at runtime)
- **Split**: chronological 80/20 — sorts by date, no shuffling, to prevent future data leaking into training

Final feature set: 16 columns (6 numeric + 10 map dummies).

## Inputs

- `data/df_featured.csv`

## Outputs

| File | Rows | Description |
|---|---|---|
| `data/X_train.csv` | 36,618 | Features — train (up to 2019-05-22) |
| `data/X_test.csv` | 9,155 | Features — test (from 2019-05-22) |
| `data/y_train.csv` | 36,618 | Target — train |
| `data/y_test.csv` | 9,155 | Target — test |

## Run

```bash
uv sync
uv run jupyter lab
```

Open `preprocessing.ipynb` and run all cells.
