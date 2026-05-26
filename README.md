# CS:GO Match Outcome Prediction — MLOps Pipeline

Predicts the winner of professional CS:GO matches using 45,773 HLTV match results (2015–2020).  
Accuracy **80.4%** — ROC AUC **0.8915**.

---

## How it works (in 30 seconds)

1. **Feature engineering** — for each match, compute 8 features from the history of every team up to that date: Elo rating, recent win rates, head-to-head record, current streak, map-specific win rate.
2. **Train** — a sklearn `Pipeline` (StandardScaler + OneHotEncoder + XGBoost) trained on 80% of matches, chronologically ordered.
3. **Inference** — given two team names and a map, replay the full match history to reconstruct current team states, compute features, and predict.

---

## Quick start

```bash
# CLI inference
uv run python predict.py --team1 NaVi --team2 Astralis --map Dust2
#   → Astralis wins  (96.8% confidence)

# REST API
uv run python serve.py
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"team1": "NaVi", "team2": "Astralis", "map": "Dust2"}'

# Full pipeline locally (skips unchanged stages)
dvc repro
```

---

## Pipeline steps

| Step | What it does |
|------|--------------|
| `01_feature_engineering` | Builds Elo, win rates, H2H, streak, map win rate — statefully, no leakage |
| `02_preprocessing` | Chronological 80/20 split — no shuffle to avoid temporal leakage |
| `03_train_model_kubernetes` | sklearn Pipeline: StandardScaler + OneHotEncoder + XGBClassifier |
| `04_evaluate_model` | Accuracy, ROC AUC, confusion matrix, logs to MLflow |
| `05_upload_model` | Registers model in MLflow Model Registry, promotes to Staging |
| `06_deploy_inference_service` | Deploys as KServe `InferenceService` on Kubernetes |
| `07_test_inference_service` | Smoke-tests the live endpoint |
| `08_hyperparameter_tuning` | Optuna search with `TimeSeriesSplit` CV |
| `09_model_monitoring` | Evidently drift detection + rolling accuracy |

---

## Run locally

### Prerequisites

- Python 3.11+ and [uv](https://github.com/astral-sh/uv): `pip install uv`

### Option A — One command (DVC)

DVC runs only stages whose inputs changed:

```bash
dvc repro           # run only what changed
dvc repro --force   # re-run everything
```

### Option B — Step by step

```bash
# 1. Feature engineering — builds 8 features per match from historical state
cd 01_feature_engineering && uv sync && uv run python feature_engineering.py
# → data/df_featured.csv

# 2. Preprocessing — chronological 80/20 split, no shuffle
cd 02_preprocessing && uv sync && uv run python preprocessing.py
# → data/X_train.csv, X_test.csv, y_train.csv, y_test.csv

# (optional) Hyperparameter tuning — run before step 3
cd 08_hyperparameter_tuning && uv sync && uv run python hyperparameter_tuning.py
# → data/best_params.json  (picked up automatically by train.py)

# 3. Train — fits StandardScaler + OneHotEncoder + XGBoost in one Pipeline
cd 03_train_model_kubernetes && uv sync
DATA_DIR=$(pwd)/../data MODEL_DIR=$(pwd)/../data MODEL_TYPE=xgboost uv run python train.py
# → data/model.pkl

# 4. Evaluate
cd 04_evaluate_model && uv sync && uv run python evaluate_model.py
# → data/evaluation.json, data/confusion_matrix.png

# 5. Upload to MLflow registry
cd 05_upload_model && uv sync && uv run python upload_model.py

# 6. Deploy on Kubernetes (KServe)
kubectl apply -f 06_deploy_inference_service/inference_service.yaml

# 7. Test the live endpoint
cd 07_test_inference_service
cp .env.example .env  # set INFERENCE_URL
uv sync && uv run python test_inference_service.py

# 9. Monitoring — drift detection + rolling accuracy
cd 09_model_monitoring && uv sync && uv run python model_monitoring.py
```

---

## Run on Kubeflow

No Docker build required. The pipeline runs using `python:3.11-slim` as base image and installs
packages at runtime. `results.csv` is downloaded from GitHub by the first step.
Intermediate files (CSVs, model) are shared between steps via a PVC.

### 1 — Create the shared volume

```bash
kubectl apply -f pvc.yaml
```

This creates a 2Gi `ReadWriteMany` PVC named `csgo-data-pvc` in the `kubeflow-user-example-com`
namespace, using the `nfs-rwx` storage class. All pipeline Pods mount it at `/data`.

### 2 — Compile the pipeline

```bash
uv run python pipeline.py
# → pipeline.yaml
```

### 3 — Test the TrainJob in isolation

Before running the full pipeline, you can test the training step alone (uploads data to SeaweedFS, submits a TrainJob, waits for completion, downloads the model):

```bash
# In a separate terminal — expose SeaweedFS locally
kubectl port-forward -n kubeflow svc/seaweedfs 9000:8333

# Run the test (default timeout 700s — the job takes ~8 min due to Istio init + pip install)
uv run python test_trainjob.py

# Override namespace or timeout
uv run python test_trainjob.py --namespace csgo --wait-timeout 900
```

Expected output:
```
1. Upload des données de test sur SeaweedFS...
  uploaded → s3://mlpipeline/csgo/test-data/<ts>/X_train.csv
  uploaded → s3://mlpipeline/csgo/test-data/<ts>/y_train.csv

2. Soumission du TrainJob...
TrainJob 'csgo-train-test-<ts>' soumis dans 'csgo'.

3. Attente...
  [  0s] active=1 succeeded=0 failed=0
  ...
TrainJob terminé avec succès.

4. Téléchargement du modèle...
  Modèle chargé : Pipeline
Test OK.
```

Follow logs live while the job runs:
```bash
kubectl logs -n csgo -l trainer.kubeflow.org/trainjob=csgo-train-test-<ts> -f
```

> **Note:** The job runs on a GPU node (`mlops-worker-3` or `mlops-worker-4`). Istio sidecar injection adds ~4 minutes of init time before training starts.

### 5 — Upload and run

```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
#ids ==> csgo@example.com : ***
```

Open `http://localhost:8080`, then:

- **Pipelines** → **Upload pipeline** → select `pipeline.yaml`
- **Create run** → leave default parameters → **Start**

### 6 — What each step does on the cluster

```
feature-engineering   downloads results.csv from GitHub → computes 8 features → writes df_featured.csv to PVC
preprocessing         reads df_featured.csv from PVC → 80/20 split → writes X_train, X_test, y_train, y_test
train                 reads X_train from PVC → fits XGBoost Pipeline → writes model.pkl to PVC
evaluate              reads model.pkl + X_test from PVC → prints accuracy & AUC → writes evaluation.json
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `xgboost` | `xgboost` or `logreg` |
| `train_ratio` | `0.8` | Train/test split ratio |

---

## Inference

### CLI

```bash
uv run python predict.py --team1 NaVi --team2 Astralis --map Dust2
```

Replays all 45,773 matches to build current team states, then predicts. ~1–2s.

### REST API

```bash
uv run python serve.py   # state built once at startup, served in-memory
```

```
POST /predict   {"team1": "NaVi", "team2": "Astralis", "map": "Dust2"}
GET  /health    {"status": "ok", "teams_in_history": 1554}
```

---

## Dataset

Source: HLTV professional CS:GO match results (2015-11-03 → 2020-03-18). Git-tracked.

| File | Rows | Description |
|------|------|-------------|
| `data/results.csv` | 45,773 | Raw match results |
| `data/df_featured.csv` | 45,773 | + 8 features + target |
| `data/X_train.csv` | 36,618 | Train (up to 2019-05-22) |
| `data/X_test.csv` | 9,155 | Test (from 2019-05-22) |

**Target**: `team_1_wins` — 1 if team_1 won, 0 otherwise. Class balance ≈ 0.54 / 0.46.

---

## Features

All features are differences (team_1 − team_2), computed **before** each match to prevent leakage.

| Feature | Description |
|---------|-------------|
| `elo_diff` | Elo rating difference, K=32 |
| `winrate_10_diff` | Win rate over last 10 matches |
| `winrate_30_diff` | Win rate over last 30 matches |
| `experience_diff` | Total matches played |
| `rank_diff` | HLTV rank (lower = better) |
| `h2h_winrate` | Head-to-head win rate of team_1 vs team_2 |
| `streak_diff` | Current win/loss streak (+3 = team_1 on 3-win streak) |
| `map_winrate_diff` | Win rate on the specific map being played |
| `_map` | Map name — one-hot encoded inside the Pipeline |

---

## Model

`sklearn.pipeline.Pipeline` — self-contained, no separate preprocessing at inference.

```
ColumnTransformer
  ├── StandardScaler    → 8 numeric features
  └── OneHotEncoder     → _map (10 possible values)
        ↓
  XGBClassifier (n_estimators=300, learning_rate=0.05, max_depth=6)
```

| Metric | XGBoost | LogisticRegression |
|--------|---------|--------------------|
| Test accuracy | **0.8043** | 0.769 |
| ROC AUC | **0.8915** | 0.851 |

---

## Repository layout

```
.
├── data/                          # datasets + model (git-tracked)
├── params.yaml                    # DVC pipeline hyperparameters
├── dvc.yaml                       # DVC pipeline DAG (local runs)
├── pipeline.py                    # Kubeflow Pipelines v2 DAG
├── pipeline.yaml                  # compiled KFP pipeline — upload this to the UI
├── pvc.yaml                       # Kubernetes PVC for inter-step data sharing
├── predict.py                     # CLI inference: --team1 X --team2 Y --map Z
├── serve.py                       # FastAPI REST API
├── Dockerfile                     # image for all steps (optional, not needed for KFP)
├── 01_feature_engineering/
├── 02_preprocessing/
├── 03_train_model_kubernetes/
│   ├── train.py
│   └── job.yaml                   # standalone Kubernetes Job (without KFP)
├── 04_evaluate_model/
├── 05_upload_model/
├── 06_deploy_inference_service/
├── 07_test_inference_service/
├── 08_hyperparameter_tuning/
└── 09_model_monitoring/
```
