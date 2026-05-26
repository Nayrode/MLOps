"""
Standalone Kubeflow Training Job — CS:GO match predictor.

Soumet un TrainJob via le Kubeflow Training SDK (v2).
À appeler avant la pipeline KFP pour entraîner le modèle.

Usage:
    uv run python kubeflow_train.py
    uv run python kubeflow_train.py --namespace csgo --wait
"""
import argparse
from kubeflow.training import TrainingClient


def train_func(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    train_ratio: float = 0.8,
    k_elo: int = 32,
    s3_endpoint: str = "http://seaweedfs.kubeflow.svc.cluster.local:8333",
    s3_bucket: str = "mlpipeline",
    s3_model_key: str = "csgo/models/model.pkl",
):
    """
    Fonction de training exécutée dans le container Kubeflow.
    Télécharge les données, fait le feature engineering, entraîne et sauvegarde le modèle.
    """
    import io
    import os
    import urllib.request

    import boto3
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    REPO = "https://raw.githubusercontent.com/Nayrode/MLOps/main"
    NUMERIC_COLS = [
        "elo_diff", "winrate_10_diff", "winrate_30_diff",
        "experience_diff", "rank_diff", "h2h_winrate",
        "streak_diff", "map_winrate_diff",
    ]
    FEATURE_COLS = NUMERIC_COLS + ["_map"]

    # ── Download raw data ─────────────────────────────────────────────────────
    raw_path = "/tmp/results.csv"
    urllib.request.urlretrieve(f"{REPO}/data/results.csv", raw_path)
    df = pd.read_csv(raw_path, index_col=0, parse_dates=["date"])
    print(f"Downloaded {len(df)} rows")

    # ── Feature engineering ───────────────────────────────────────────────────
    def _streak(history):
        if not history:
            return 0
        val = history[-1]
        streak = 0
        for r in reversed(history):
            if r == val:
                streak += 1
            else:
                break
        return streak if val == 1 else -streak

    teams = pd.concat([df["team_1"], df["team_2"]]).unique()
    state = {
        "elo":            {t: 1500 for t in teams},
        "matches_played": {t: 0    for t in teams},
        "win_history":    {t: []   for t in teams},
        "h2h":            {},
        "map_history":    {t: {}   for t in teams},
    }

    rows, labels = [], []
    for _, match in df.iterrows():
        team, opp, map_name = match["team_1"], match["team_2"], match["_map"]
        th  = state["win_history"].get(team, [])
        oh  = state["win_history"].get(opp,  [])
        h2h = state["h2h"].get((team, opp), [])
        mht = state["map_history"].get(team, {}).get(map_name, [])
        mho = state["map_history"].get(opp,  {}).get(map_name, [])

        rows.append({
            "elo_diff":         state["elo"].get(team, 1500) - state["elo"].get(opp, 1500),
            "winrate_10_diff":  (np.mean(th[-10:]) if th else 0) - (np.mean(oh[-10:]) if oh else 0),
            "winrate_30_diff":  (np.mean(th[-30:]) if th else 0) - (np.mean(oh[-30:]) if oh else 0),
            "experience_diff":  state["matches_played"].get(team, 0) - state["matches_played"].get(opp, 0),
            "rank_diff":        match["rank_1"] - match["rank_2"],
            "h2h_winrate":      np.mean(h2h) if h2h else 0.5,
            "streak_diff":      _streak(th) - _streak(oh),
            "map_winrate_diff": (np.mean(mht) if mht else 0.5) - (np.mean(mho) if mho else 0.5),
            "_map":             map_name,
        })
        labels.append(int(match["match_winner"] == 1))

        result = labels[-1]
        r_t = state["elo"].get(team, 1500)
        r_o = state["elo"].get(opp,  1500)
        exp = 1 / (1 + 10 ** ((r_o - r_t) / 400))
        state["elo"][team] = r_t + k_elo * (result - exp)
        state["elo"][opp]  = r_o + k_elo * ((1 - result) - (1 - exp))
        state["win_history"].setdefault(team, []).append(result)
        state["win_history"].setdefault(opp,  []).append(1 - result)
        state["matches_played"][team] = state["matches_played"].get(team, 0) + 1
        state["matches_played"][opp]  = state["matches_played"].get(opp,  0) + 1
        state["h2h"].setdefault((team, opp), []).append(result)
        state["map_history"].setdefault(team, {}).setdefault(map_name, []).append(result)
        state["map_history"].setdefault(opp,  {}).setdefault(map_name, []).append(1 - result)

    X = pd.DataFrame(rows)[FEATURE_COLS]
    y = pd.Series(labels)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    split = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("scaler",  StandardScaler(),                                        NUMERIC_COLS),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
        ])),
        ("classifier", XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric="logloss",
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    print(f"Train accuracy : {model.score(X_train, y_train):.4f}")

    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Test  accuracy : {acc:.4f} | ROC AUC : {auc:.4f}")

    # ── Save model to SeaweedFS (S3-compatible) ───────────────────────────────
    model_path = "/tmp/model.pkl"
    joblib.dump(model, model_path)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123"),
    )

    # Crée le bucket si nécessaire
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if s3_bucket not in existing:
        s3.create_bucket(Bucket=s3_bucket)

    s3.upload_file(model_path, s3_bucket, s3_model_key)
    print(f"Model saved → s3://{s3_bucket}/{s3_model_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="csgo")
    parser.add_argument("--name",      default="csgo-xgboost-training")
    parser.add_argument("--wait",      action="store_true", help="Attendre la fin du job")
    args = parser.parse_args()

    client = TrainingClient(namespace=args.namespace)

    client.create_job(
        name=args.name,
        train_func=train_func,
        parameters={
            "n_estimators":  300,
            "learning_rate": 0.05,
            "max_depth":     6,
            "train_ratio":   0.8,
            "k_elo":         32,
        },
        base_image="python:3.11-slim",
        packages_to_install=[
            "pandas==2.2.3",
            "numpy==2.4.6",
            "scikit-learn==1.8.0",
            "xgboost==3.2.0",
            "joblib==1.5.3",
            "boto3==1.43.14",
        ],
        num_workers=1,
    )

    print(f"TrainJob '{args.name}' soumis dans le namespace '{args.namespace}'.")
    print(f"Suivi : kubectl get trainjobs -n {args.namespace}")

    if args.wait:
        client.wait_for_job_conditions(
            name=args.name,
            expected_conditions={"Succeeded"},
        )
        print("Training terminé avec succès.")
