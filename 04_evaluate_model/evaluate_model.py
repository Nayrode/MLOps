import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

load_dotenv()

DATA_DIR            = Path("../data")
MODEL_PATH          = DATA_DIR / "model.pkl"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME     = "csgo-match-predictor"

model  = joblib.load(MODEL_PATH)
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
y_test = pd.read_csv(DATA_DIR / "y_test.csv", index_col=0).squeeze()
print(f"Model  : {type(model.named_steps['classifier']).__name__}")
print(f"X_test : {X_test.shape}")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["team_2 wins", "team_1 wins"]))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["team_2 wins", "team_1 wins"]).plot(colorbar=False)
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig(DATA_DIR / "confusion_matrix.png", dpi=120)
print(f"Saved confusion_matrix.png")

metrics = {"accuracy": accuracy, "roc_auc": roc_auc}
with open(DATA_DIR / "evaluation.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved evaluation.json: {metrics}")

import mlflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="evaluate"):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(DATA_DIR / "confusion_matrix.png"))
print("Logged to MLflow")
