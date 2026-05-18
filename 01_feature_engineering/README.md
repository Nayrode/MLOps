# 01 — Feature Engineering

Builds match-level features from raw results using stateful iteration over the match history. Features are computed strictly from data prior to each match to avoid leakage.

## Features built

| Feature | Description |
|---|---|
| `elo_diff` | Elo rating difference (team_1 − team_2) |
| `winrate_10_diff` | Win rate over last 10 matches, difference |
| `winrate_30_diff` | Win rate over last 30 matches, difference |
| `experience_diff` | Total matches played difference |
| `rank_diff` | Rank difference (team_1 − team_2) |
| `h2h_winrate` | Head-to-head win rate for team_1 vs team_2 |

All features are scaled with `StandardScaler`.

## Inputs

- `data/results.csv`

## Outputs

- `data/df_featured.csv` — original columns + 6 scaled features + `team_1_wins` target

## Run

```bash
uv sync
uv run jupyter lab
```

Open `feature_engineering.ipynb` and run all cells.
