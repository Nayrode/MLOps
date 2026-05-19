# 02 — Preprocessing

Produces a chronological train/test split ready for model training.

## What it does

- **`_map`**: kept as string — encoding is deferred to the sklearn Pipeline in step 03 (`OneHotEncoder`)
- **Team names**: skipped — 1,200+ unique values, already captured by Elo/winrate features
- **NaN handling**: none needed; dataset is fully clean (asserted at runtime)
- **Split**: chronological 80/20 — sorts by date, no shuffling, to prevent future data leaking into training

Final feature set: 7 columns (`elo_diff`, `winrate_10_diff`, `winrate_30_diff`, `experience_diff`, `rank_diff`, `h2h_winrate`, `_map`).

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
uv sync && uv run python preprocessing.py
```
