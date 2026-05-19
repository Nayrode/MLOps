from pathlib import Path

import pandas as pd

INPUT_PATH = Path("../data/df_featured.csv")
OUTPUT_DIR = Path("../data")

NUMERIC_FEATURE_COLS = [
    "elo_diff", "winrate_10_diff", "winrate_30_diff",
    "experience_diff", "rank_diff", "h2h_winrate",
    "streak_diff", "map_winrate_diff",
]
FEATURE_COLS = NUMERIC_FEATURE_COLS + ["_map"]
TARGET_COL = "team_1_wins"
TRAIN_RATIO = 0.8

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")

    assert df.isnull().sum().sum() == 0, "Unexpected NaN values found"

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    print(f"Train: {len(X_train)} rows (up to {train_df['date'].max().date()})")
    print(f"Test:  {len(X_test)} rows (from {test_df['date'].min().date()})")
    print(f"Class balance — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv")
    X_test.to_csv(OUTPUT_DIR / "X_test.csv")
    y_train.to_csv(OUTPUT_DIR / "y_train.csv")
    y_test.to_csv(OUTPUT_DIR / "y_test.csv")
    print("Saved X_train, X_test, y_train, y_test")
