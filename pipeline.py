"""Kubeflow Pipelines v2 — CS:GO match predictor.

No Docker build, no MinIO required.
Each step reads/writes to a shared NFS PVC at /data.
Steps are ordered with .after() — no KFP artifact passing needed.

Usage:
    uv run python pipeline.py              # compile → pipeline.yaml
    uv run python pipeline.py --run        # compile + submit to KFP
"""
import argparse
from kfp import compiler, dsl
from kfp.kubernetes import mount_pvc

BASE_IMAGE = "python:3.11-slim"
PVC_NAME   = "csgo-data-pvc"   # must exist in kubeflow-user-example-com namespace
DATA       = "/data"


# ── Components ───────────────────────────────────────────────────────────────

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "numpy==2.2.3"],
)
def feature_engineering():
    import numpy as np
    import pandas as pd
    from pathlib import Path

    def _streak(history):
        if not history:
            return 0
        val = history[-1]
        streak = 0
        for r in reversed(history):
            if r == val: streak += 1
            else: break
        return streak if val == 1 else -streak

    def build_state(df):
        state = {"elo": {}, "matches_played": {}, "win_history": {}, "h2h": {}, "map_history": {}}
        for team in pd.concat([df["team_1"], df["team_2"]]).unique():
            state["elo"][team] = 1500
            state["matches_played"][team] = 0
            state["win_history"][team] = []
            state["map_history"][team] = {}
        return state

    def compute_features(match, state):
        team, opp, map_name = match["team_1"], match["team_2"], match["_map"]
        th = state["win_history"].get(team, [])
        oh = state["win_history"].get(opp, [])
        h2h = state["h2h"].get((team, opp), [])
        mht = state["map_history"].get(team, {}).get(map_name, [])
        mho = state["map_history"].get(opp, {}).get(map_name, [])
        return {
            "elo_diff":        state["elo"].get(team, 1500) - state["elo"].get(opp, 1500),
            "winrate_10_diff": (np.mean(th[-10:]) if th else 0) - (np.mean(oh[-10:]) if oh else 0),
            "winrate_30_diff": (np.mean(th[-30:]) if th else 0) - (np.mean(oh[-30:]) if oh else 0),
            "experience_diff": state["matches_played"].get(team, 0) - state["matches_played"].get(opp, 0),
            "rank_diff":       match["rank_1"] - match["rank_2"],
            "h2h_winrate":     np.mean(h2h) if h2h else 0.5,
            "streak_diff":     _streak(th) - _streak(oh),
            "map_winrate_diff": (np.mean(mht) if mht else 0.5) - (np.mean(mho) if mho else 0.5),
        }

    def update_state(match, state, k=32):
        team, opp, map_name = match["team_1"], match["team_2"], match["_map"]
        result = int(match["match_winner"] == 1)
        r_t, r_o = state["elo"].get(team, 1500), state["elo"].get(opp, 1500)
        exp = 1 / (1 + 10 ** ((r_o - r_t) / 400))
        state["elo"][team] = r_t + k * (result - exp)
        state["elo"][opp]  = r_o + k * ((1 - result) - (1 - exp))
        state["win_history"].setdefault(team, []).append(result)
        state["win_history"].setdefault(opp,  []).append(1 - result)
        state["matches_played"][team] = state["matches_played"].get(team, 0) + 1
        state["matches_played"][opp]  = state["matches_played"].get(opp,  0) + 1
        state["h2h"].setdefault((team, opp), []).append(result)
        state["map_history"].setdefault(team, {}).setdefault(map_name, []).append(result)
        state["map_history"].setdefault(opp,  {}).setdefault(map_name, []).append(1 - result)
        return state

    import urllib.request
    url = "https://raw.githubusercontent.com/Nayrode/MLOps/main/data/results.csv"
    urllib.request.urlretrieve(url, f"{DATA}/results.csv")

    df = pd.read_csv(f"{DATA}/results.csv", index_col=0, parse_dates=["date"])
    print(f"Loaded {len(df)} rows")
    state = build_state(df)
    X, y = [], []
    for _, match in df.iterrows():
        X.append(compute_features(match, state))
        y.append(int(match["match_winner"] == 1))
        state = update_state(match, state)
    out = pd.concat([df, pd.DataFrame(X), pd.Series(y, name="team_1_wins")], axis=1)
    out.to_csv(f"{DATA}/df_featured.csv")
    print(f"Saved {len(out)} rows → {DATA}/df_featured.csv")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3"],
)
def preprocessing(train_ratio: float = 0.8):
    import pandas as pd

    FEATURE_COLS = ["elo_diff", "winrate_10_diff", "winrate_30_diff",
                    "experience_diff", "rank_diff", "h2h_winrate",
                    "streak_diff", "map_winrate_diff", "_map"]

    df = pd.read_csv(f"{DATA}/df_featured.csv", index_col=0, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    train_df[FEATURE_COLS].to_csv(f"{DATA}/X_train.csv")
    train_df["team_1_wins"].to_csv(f"{DATA}/y_train.csv")
    test_df[FEATURE_COLS].to_csv(f"{DATA}/X_test.csv")
    test_df["team_1_wins"].to_csv(f"{DATA}/y_test.csv")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1", "xgboost==3.2.0", "joblib==1.5.0"],
)
def train(model_type: str = "xgboost"):
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    NUMERIC_COLS = ["elo_diff", "winrate_10_diff", "winrate_30_diff",
                    "experience_diff", "rank_diff", "h2h_winrate",
                    "streak_diff", "map_winrate_diff"]

    X = pd.read_csv(f"{DATA}/X_train.csv", index_col=0)
    y = pd.read_csv(f"{DATA}/y_train.csv", index_col=0).squeeze()

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("scaler",  StandardScaler(), NUMERIC_COLS),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
        ])),
        ("classifier", XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, eval_metric="logloss", random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    print(f"Train accuracy: {pipeline.score(X, y):.4f}")
    joblib.dump(pipeline, f"{DATA}/model.pkl")
    print(f"Saved model → {DATA}/model.pkl")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1", "xgboost==3.2.0", "joblib==1.5.0"],
)
def evaluate():
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    import json

    pipeline = joblib.load(f"{DATA}/model.pkl")
    X = pd.read_csv(f"{DATA}/X_test.csv",  index_col=0)
    y = pd.read_csv(f"{DATA}/y_test.csv",  index_col=0).squeeze()

    acc = accuracy_score(y, pipeline.predict(X))
    auc = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    with open(f"{DATA}/evaluation.json", "w") as f:
        json.dump({"accuracy": acc, "roc_auc": auc}, f, indent=2)


# ── Pipeline DAG ─────────────────────────────────────────────────────────────

@dsl.pipeline(name="csgo-match-predictor")
def csgo_pipeline(model_type: str = "xgboost", train_ratio: float = 0.8):

    fe  = feature_engineering()
    pre = preprocessing(train_ratio=train_ratio).after(fe)
    tr  = train(model_type=model_type).after(pre)
    ev  = evaluate().after(tr)

    for task in [fe, pre, tr, ev]:
        mount_pvc(task, pvc_name=PVC_NAME, mount_path=DATA)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  action="store_true")
    parser.add_argument("--host", default="http://localhost:3000")
    args = parser.parse_args()

    compiler.Compiler().compile(csgo_pipeline, "pipeline.yaml")
    print("Compiled → pipeline.yaml")

    if args.run:
        import kfp
        client = kfp.Client(host=args.host)
        run = client.create_run_from_pipeline_func(
            csgo_pipeline,
            arguments={"model_type": "xgboost"},
            run_name="csgo-run",
        )
        print(f"Submitted → {run.run_id}")
