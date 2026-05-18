#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "scikit-learn",
# ]
# ///

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_feature_state(df, base_elo=1500):
    state = {
        "elo": {},
        "matches_played": {},
        "win_history": {},
        "h2h": {},
    }
    teams = pd.concat([df["team_1"], df["team_2"]]).unique()
    for team in teams:
        state["elo"][team] = base_elo
        state["matches_played"][team] = 0
        state["win_history"][team] = []
    return state


def compute_features_for_match(match, state):
    team = match["team_1"]
    opp = match["team_2"]

    elo_team = state["elo"].get(team, 1500)
    elo_opp = state["elo"].get(opp, 1500)
    elo_diff = elo_team - elo_opp

    winrate_10 = (
        np.mean(state["win_history"].get(team, [])[-10:])
        if len(state["win_history"].get(team, [])) > 0
        else 0
    )
    winrate_30 = (
        np.mean(state["win_history"].get(team, [])[-30:])
        if len(state["win_history"].get(team, [])) > 0
        else 0
    )
    winrate_10_diff = winrate_10 - (
        np.mean(state["win_history"].get(opp, [])[-10:])
        if len(state["win_history"].get(opp, [])) > 0
        else 0
    )
    winrate_30_diff = winrate_30 - (
        np.mean(state["win_history"].get(opp, [])[-30:])
        if len(state["win_history"].get(opp, [])) > 0
        else 0
    )

    experience_diff = state["matches_played"].get(team, 0) - state[
        "matches_played"
    ].get(opp, 0)

    rank_diff = match["rank_1"] - match["rank_2"]

    h2h_key = (team, opp)
    h2h_list = state["h2h"].get(h2h_key, [])
    h2h_winrate = np.mean(h2h_list) if len(h2h_list) > 0 else 0.5

    return {
        "elo_diff": elo_diff,
        "winrate_10_diff": winrate_10_diff,
        "winrate_30_diff": winrate_30_diff,
        "experience_diff": experience_diff,
        "rank_diff": rank_diff,
        "h2h_winrate": h2h_winrate,
    }


def update_state_with_result(match, state, k=32):
    team = match["team_1"]
    opp = match["team_2"]
    result = int(match["match_winner"] == 1)

    r_team = state["elo"].get(team, 1500)
    r_opp = state["elo"].get(opp, 1500)
    exp = 1 / (1 + 10 ** ((r_opp - r_team) / 400))
    state["elo"][team] = r_team + k * (result - exp)
    state["elo"][opp] = r_opp + k * ((1 - result) - (1 - exp))

    state["win_history"].setdefault(team, []).append(result)
    state["win_history"].setdefault(opp, []).append(1 - result)

    state["matches_played"][team] = state["matches_played"].get(team, 0) + 1
    state["matches_played"][opp] = state["matches_played"].get(opp, 0) + 1

    h2h_key = (team, opp)
    state["h2h"].setdefault(h2h_key, []).append(result)
    return state


def match_to_features(preprocessor, history_df, match):
    """Compute scaled features for a single future match given historical data."""
    past = history_df[history_df["date"] < match["date"]]
    state = build_feature_state(past)
    for _, m in past.sort_values("date").iterrows():
        update_state_with_result(m, state)
    X = pd.DataFrame([compute_features_for_match(match, state)])
    return preprocessor.transform(X)


def build_features(df):
    state = build_feature_state(df)
    X, y = [], []
    for _, match in df.iterrows():
        X.append(compute_features_for_match(match, state))
        y.append(int(match["match_winner"] == 1))
        state = update_state_with_result(match, state)
    return pd.concat(
        [df, pd.DataFrame(X), pd.Series(y, name="team_1_wins")], axis=1
    )


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for match data")
    parser.add_argument(
        "--input", default="../data/results.csv", help="Path to raw CSV"
    )
    parser.add_argument(
        "--output", default="../data/df_featured.csv", help="Path to output CSV"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, index_col=0, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {input_path}")

    df_featured = build_features(df)

    feature_cols = [
        "elo_diff", "winrate_10_diff", "winrate_30_diff",
        "experience_diff", "rank_diff", "h2h_winrate",
    ]
    scaler = StandardScaler()
    df_featured[feature_cols] = scaler.fit_transform(df_featured[feature_cols])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(output_path)
    print(f"Saved {len(df_featured)} rows to {output_path}")
    print(df_featured[feature_cols + ["team_1_wins"]].describe())


if __name__ == "__main__":
    main()
