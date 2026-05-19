import argparse
import joblib
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Predict CS:GO match outcome")
parser.add_argument("--elo-diff",        type=float, required=True)
parser.add_argument("--winrate-10-diff", type=float, required=True)
parser.add_argument("--winrate-30-diff", type=float, required=True)
parser.add_argument("--experience-diff", type=float, required=True)
parser.add_argument("--rank-diff",       type=float, required=True)
parser.add_argument("--h2h-winrate",     type=float, required=True)
parser.add_argument("--map",             type=str,   required=True,
                    choices=["Cache","Cobblestone","Default","Dust2",
                             "Inferno","Mirage","Nuke","Overpass","Train","Vertigo"])
args = parser.parse_args()

pipeline = joblib.load(Path(__file__).parent / "data/model.pkl")

match = pd.DataFrame([{
    "elo_diff":        args.elo_diff,
    "winrate_10_diff": args.winrate_10_diff,
    "winrate_30_diff": args.winrate_30_diff,
    "experience_diff": args.experience_diff,
    "rank_diff":       args.rank_diff,
    "h2h_winrate":     args.h2h_winrate,
    "_map":            args.map,
}])

proba = pipeline.predict_proba(match)[0][1]
winner = "team_1" if proba >= 0.5 else "team_2"
print(f"{winner} wins — {proba:.1%} confidence")
