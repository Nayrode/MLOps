from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("../data/results.csv")
OUTPUT_PATH = Path("../data/df_featured.csv")


def build_feature_state(df, base_elo=1500):
    state = {"elo": {}, "matches_played": {}, "win_history": {}, "h2h": {}}
    for team in pd.concat([df["team_1"], df["team_2"]]).unique():
        state["elo"][team] = base_elo
        state["matches_played"][team] = 0
        state["win_history"][team] = []
    return state


def compute_features_for_match(match, state):
    team = match["team_1"]
    opp = match["team_2"]

    team_history = state["win_history"].get(team, [])
    opp_history = state["win_history"].get(opp, [])

    winrate_10 = np.mean(team_history[-10:]) if team_history else 0
    winrate_30 = np.mean(team_history[-30:]) if team_history else 0
    h2h_list = state["h2h"].get((team, opp), [])

    return {
        "elo_diff": state["elo"].get(team, 1500) - state["elo"].get(opp, 1500),
        "winrate_10_diff": winrate_10 - (np.mean(opp_history[-10:]) if opp_history else 0),
        "winrate_30_diff": winrate_30 - (np.mean(opp_history[-30:]) if opp_history else 0),
        "experience_diff": state["matches_played"].get(team, 0) - state["matches_played"].get(opp, 0),
        "rank_diff": match["rank_1"] - match["rank_2"],
        "h2h_winrate": np.mean(h2h_list) if h2h_list else 0.5,
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
    state["h2h"].setdefault((team, opp), []).append(result)
    return state


def build_features(df):
    state = build_feature_state(df)
    X, y = [], []
    for _, match in df.iterrows():
        X.append(compute_features_for_match(match, state))
        y.append(int(match["match_winner"] == 1))
        state = update_state_with_result(match, state)
    return pd.concat([df, pd.DataFrame(X), pd.Series(y, name="team_1_wins")], axis=1)


def match_to_features(history_df, match):
    """Return raw feature row for a single future match given historical data."""
    past = history_df[history_df["date"] < match["date"]].sort_values("date")
    state = build_feature_state(past)
    for _, m in past.iterrows():
        update_state_with_result(m, state)
    features = compute_features_for_match(match, state)
    features["_map"] = match["_map"]
    return pd.DataFrame([features])


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    df_featured = build_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(OUTPUT_PATH)
    print(f"Saved {len(df_featured)} rows to {OUTPUT_PATH}")
