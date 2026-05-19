"""Training script — runs inside the Kubernetes Job container."""
from pathlib import Path
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR  = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
EXPERIMENT_NAME     = os.environ.get("MLFLOW_EXPERIMENT", "csgo-match-predictor")

NUMERIC_COLS = [
    "elo_diff", "winrate_10_diff", "winrate_30_diff",
    "experience_diff", "rank_diff", "h2h_winrate",
]

best_params_path = DATA_DIR / "best_params.json"
if best_params_path.exists():
    with open(best_params_path) as f:
        params = json.load(f)["params"]
    print(f"Loaded best params: {params}")
else:
    params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    print(f"No best_params.json — using defaults: {params}")

X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col=0).squeeze()
print(f"Loaded {len(X_train)} rows, {X_train.shape[1]} features")

pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("scaler",  StandardScaler(), NUMERIC_COLS),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
    ])),
    ("classifier", LogisticRegression(**params)),
])

pipeline.fit(X_train, y_train)
train_acc = pipeline.score(X_train, y_train)
print(f"Train accuracy : {train_acc:.4f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / "model.pkl")
print(f"Pipeline saved : {MODEL_DIR / 'model.pkl'}")

if MLFLOW_TRACKING_URI:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print(f"Run ID         : {run.info.run_id}")
