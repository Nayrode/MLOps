# 01 — Feature Engineering

Builds match-level features from raw results using stateful iteration over the match history. Features are computed strictly from data prior to each match to avoid leakage.

## Features built

| Feature | Description |
|---------|-------------|
| `elo_diff` | Elo rating difference (team_1 − team_2), K=32 |
| `winrate_10_diff` | Win rate over last 10 matches, difference |
| `winrate_30_diff` | Win rate over last 30 matches, difference |
| `experience_diff` | Total matches played difference |
| `rank_diff` | HLTV rank difference (team_1 − team_2) |
| `h2h_winrate` | Head-to-head win rate for team_1 vs team_2 |
| `streak_diff` | Current win/loss streak difference (positive = team_1 on win streak) |
| `map_winrate_diff` | Win rate difference specifically on the played map |

Features are **unscaled** — scaling happens inside the sklearn Pipeline in step 03.

## How it works

State (Elo, win history, H2H, map history) is updated after each match in chronological order. Each row's features are computed from the state **before** that match is processed, so no future information leaks.

## Inputs

- `data/results.csv`

## Outputs

- `data/df_featured.csv` — original columns + 8 features + `team_1_wins` target

## Run

```bash
uv sync && uv run python feature_engineering.py
```
