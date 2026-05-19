"""Training script — runs inside the Kubernetes Job container."""
from pathlib import Path
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_DIR  = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
EXPERIMENT_NAME     = os.environ.get("MLFLOW_EXPERIMENT", "csgo-match-predictor")
C        = float(os.environ.get("C", "1.0"))
MAX_ITER = int(os.environ.get("MAX_ITER", "1000"))
SOLVER   = os.environ.get("SOLVER", "lbfgs")

# Override with best params from step 08 if present
best_params_path = DATA_DIR / "best_params.json"
if best_params_path.exists():
    with open(best_params_path) as f:
        best = json.load(f)["params"]
    C        = float(best.get("C", C))
    MAX_ITER = int(best.get("max_iter", MAX_ITER))
    SOLVER   = best.get("solver", SOLVER)
    print(f"Loaded best params: C={C}, max_iter={MAX_ITER}, solver={SOLVER}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col=0).squeeze()
print(f"Loaded {len(X_train)} rows, {X_train.shape[1]} features")

with mlflow.start_run(run_name="train") as run:
    mlflow.log_params({"C": C, "max_iter": MAX_ITER, "solver": SOLVER, "model": "LogisticRegression"})

    model = LogisticRegression(C=C, max_iter=MAX_ITER, solver=SOLVER)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.sklearn.log_model(model, artifact_path="model")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.pkl")

    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Run ID         : {run.info.run_id}")
    print(f"Model saved    : {MODEL_DIR / 'model.pkl'}")
