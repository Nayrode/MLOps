from pathlib import Path
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, roc_auc_score

load_dotenv()

DATA_DIR        = Path("../data")
REPORT_DIR      = DATA_DIR / "reports"
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.20"))

REPORT_DIR.mkdir(parents=True, exist_ok=True)

reference = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
current   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col=0)
y_test    = pd.read_csv(DATA_DIR / "y_test.csv",  index_col=0).squeeze()

print(f"Reference (train) : {len(reference)} rows")
print(f"Current   (test)  : {len(current)} rows")

snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)

report_path = REPORT_DIR / "drift_report.html"
snapshot.save_html(str(report_path))
print(f"Report saved to {report_path}")

metrics      = snapshot.dict()["metrics"]
drift_result = metrics[0]["value"]          # DriftedColumnsCount is always first
share_drifted = drift_result["share"]
n_drifted     = int(drift_result["count"])
n_total       = len(reference.columns)

print(f"Drifted columns: {n_drifted}/{n_total} ({share_drifted:.1%})")
if share_drifted > DRIFT_THRESHOLD:
    print(f"ALERT: drift {share_drifted:.1%} exceeds threshold {DRIFT_THRESHOLD:.1%}")
else:
    print("No significant drift detected.")

model  = joblib.load(DATA_DIR / "model.pkl")
y_pred = model.predict(current)
y_prob = model.predict_proba(current)[:, 1]
acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_prob)

print(f"\nPerformance on current data:")
print(f"  Accuracy : {acc:.4f}")
print(f"  ROC AUC  : {auc:.4f}")
