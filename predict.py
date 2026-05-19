import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "01_feature_engineering"))
from feature_engineering import build_feature_state, compute_features_for_match, update_state_with_result

MAPS = ["Cache", "Cobblestone", "Default", "Dust2",
        "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"]

parser = argparse.ArgumentParser(description="Predict CS:GO match outcome from team names")
parser.add_argument("--team1", required=True, help="Name of team 1")
parser.add_argument("--team2", required=True, help="Name of team 2")
parser.add_argument("--map",   required=True, choices=MAPS)
args = parser.parse_args()

# Load match history to compute features from context
history = pd.read_csv(ROOT / "data/results.csv", parse_dates=["date"])

# Look up each team's most recent rank
def latest_rank(df, team):
    rows = df[(df["team_1"] == team) | (df["team_2"] == team)].sort_values("date")
    if rows.empty:
        return 50  # unknown team — neutral rank
    last = rows.iloc[-1]
    return last["rank_1"] if last["team_1"] == team else last["rank_2"]

rank1 = latest_rank(history, args.team1)
rank2 = latest_rank(history, args.team2)

# Build feature state from full history then compute features for this match
state = build_feature_state(history)
for _, m in history.sort_values("date").iterrows():
    update_state_with_result(m, state)

match = {"team_1": args.team1, "team_2": args.team2,
         "rank_1": rank1, "rank_2": rank2, "_map": args.map}

features = compute_features_for_match(match, state)
features["_map"] = args.map
X = pd.DataFrame([features])

pipeline = joblib.load(ROOT / "data/model.pkl")
proba = pipeline.predict_proba(X)[0][1]
loser  = args.team2 if proba >= 0.5 else args.team1
winner = args.team1 if proba >= 0.5 else args.team2

print(f"\n  {args.team1} vs {args.team2} — {args.map}")
print(f"  → {winner} wins  ({proba if proba >= 0.5 else 1-proba:.1%} confidence)\n")
