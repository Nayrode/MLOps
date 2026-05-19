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

# Full pipeline (skips unchanged stages)
dvc repro
```

---

## Pipeline

```
01_feature_engineering
        ↓
02_preprocessing
        ↓
08_hyperparameter_tuning  ← optional, run before 03
        ↓
03_train_model_kubernetes
        ↓
04_evaluate_model
        ↓
05_upload_model  (MLflow registry)
        ↓
06_deploy_inference_service  (KServe)
        ↓
07_test_inference_service
        ↓
09_model_monitoring
```

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

## Run locally (step by step)

### Prerequisites

- Python 3.11+ and [uv](https://github.com/astral-sh/uv): `pip install uv`
- Docker + kubectl for Kubernetes steps

### Step 01 — Feature engineering

```bash
cd 01_feature_engineering && uv sync && uv run python feature_engineering.py
```

Reads `data/results.csv`, iterates matches chronologically, builds state (Elo, histories),
computes 8 features per match. Writes `data/df_featured.csv`.

### Step 02 — Preprocessing

```bash
cd 02_preprocessing && uv sync && uv run python preprocessing.py
```

Sorts by date, splits 80/20 without shuffling. Writes `X_train`, `X_test`, `y_train`, `y_test`.  
`_map` stays as a string — the sklearn Pipeline in step 03 encodes it.

### Step 08 — Hyperparameter tuning _(optional, before step 03)_

```bash
cd 08_hyperparameter_tuning && uv sync && uv run python hyperparameter_tuning.py
```

Writes `data/best_params.json`. Step 03 picks it up automatically if `model_type` matches.

### Step 03 — Train

```bash
cd 03_train_model_kubernetes && uv sync
DATA_DIR=$(pwd)/../data MODEL_DIR=$(pwd)/../data MODEL_TYPE=xgboost uv run python train.py
```

Fits the full Pipeline on `X_train`. Writes `data/model.pkl` — the serialised Pipeline
contains the scaler, encoder, and classifier. No separate transform step needed at inference.

### Step 04 — Evaluate

```bash
cd 04_evaluate_model && uv sync && uv run python evaluate_model.py
```

Loads `model.pkl` + `X_test`/`y_test`. Prints accuracy and AUC, saves `evaluation.json`
and `confusion_matrix.png`, logs everything to MLflow.

### Step 05 — Upload model

```bash
cd 05_upload_model && uv sync && uv run python upload_model.py
```

### Step 06 — Deploy inference service

```bash
cd 06_deploy_inference_service
kubectl apply -f inference_service.yaml && kubectl get inferenceservice
```

### Step 07 — Test inference service

```bash
cd 07_test_inference_service
cp .env.example .env  # set INFERENCE_URL
uv sync && uv run python test_inference_service.py
```

### Step 09 — Model monitoring

```bash
cd 09_model_monitoring && uv sync && uv run python model_monitoring.py
```

Compares feature distributions between train (reference) and test (current) with Evidently.
Alerts if more than 20% of features have drifted. Also prints rolling accuracy + AUC.

---

## Run with DVC

DVC tracks checksums of every input file and parameter. It only re-runs a stage if
something it depends on has changed.

```bash
dvc repro           # run only what changed
dvc repro --force   # re-run everything
```

Stages mirror the pipeline above: `feature_engineering → preprocessing → train → evaluate → monitoring`.  
Config is in `dvc.yaml`; hyperparameters are in `params.yaml`.

---

## Run on Kubeflow

Each step runs as an isolated Pod on Kubernetes, reading and writing to a shared PVC.
`pipeline.py` defines the DAG using the KFP v2 SDK.

### 1 — Install Kubeflow Pipelines

```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.2.0"
# UI available at:
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 3000:80
```

### 2 — Build and push the Docker image

```bash
docker build -t YOUR_REGISTRY/csgo-mlops:latest .
docker push YOUR_REGISTRY/csgo-mlops:latest
```

Then update `IMAGE` in `pipeline.py`.

### 3 — Create the shared volume and load data

```bash
kubectl apply -f pvc.yaml
# Copy raw data onto the PVC (one-time)
kubectl cp data/results.csv kubeflow/<init-pod>:/data/results.csv
```

### 4 — Compile and submit

```bash
pip install kfp
python pipeline.py          # compiles → pipeline.yaml
python pipeline.py --run    # compiles + submits to KFP at localhost:3000
```

The pipeline appears in the Kubeflow UI. Each box in the DAG is a Pod running one step.
Steps share data through the PVC (`/data` inside each container = `data/` locally).

### How `pipeline.py` maps to the scripts

```
KFP Component          Docker command
─────────────────────────────────────────────────────────────────────
feature-engineering  → python 01_feature_engineering/feature_engineering.py
preprocessing        → python 02_preprocessing/preprocessing.py
train                → DATA_DIR=/data MODEL_TYPE=xgboost python 03_train_model_kubernetes/train.py
evaluate             → python 04_evaluate_model/evaluate_model.py
monitoring           → python 09_model_monitoring/model_monitoring.py
```

No code changes needed in the scripts — they already read/write from `DATA_DIR`.

---

## Inference

### CLI

```bash
uv run python predict.py --team1 NaVi --team2 Astralis --map Dust2
```

Replays all 45,773 matches to build current team states, then predicts.  
~1–2s (state is rebuilt from scratch each call).

### REST API

```bash
uv run python serve.py   # state built once at startup, then served in-memory
```

```
POST /predict   {"team1": "NaVi", "team2": "Astralis", "map": "Dust2"}
GET  /health    {"status": "ok", "teams_in_history": 1554}
```

`serve.py` is the container image for a KServe `InferenceService` in production.

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

All features are differences (team_1 − team_2) and are computed **before** each match is processed.

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
  ├── StandardScaler       → 8 numeric features
  └── OneHotEncoder        → _map (10 possible values)
        ↓
  XGBClassifier (n_estimators=300, learning_rate=0.05, max_depth=6)
```

| Metric | XGBoost | LogisticRegression |
|--------|---------|--------------------|
| Test accuracy | **0.8043** | 0.769 |
| ROC AUC | **0.8915** | 0.851 |

Switch model: `MODEL_TYPE=logreg uv run python train.py`

---

## Repository layout

```
.
├── data/                            # datasets + model (git-tracked)
├── params.yaml                      # pipeline hyperparameters (DVC reads this)
├── dvc.yaml                         # DVC pipeline DAG
├── pipeline.py                      # Kubeflow Pipelines v2 DAG
├── pvc.yaml                         # Kubernetes PersistentVolumeClaim for KFP
├── Dockerfile                       # single image for all pipeline steps
├── predict.py                       # CLI: --team1 X --team2 Y --map Z
├── serve.py                         # FastAPI REST API (KServe image)
├── 01_feature_engineering/
├── 02_preprocessing/
├── 03_train_model_kubernetes/
│   ├── train.py
│   ├── Dockerfile                   # step-specific image (legacy, use root Dockerfile)
│   └── job.yaml                     # standalone Kubernetes Job (without KFP)
├── 04_evaluate_model/
├── 05_upload_model/
├── 06_deploy_inference_service/
├── 07_test_inference_service/
├── 08_hyperparameter_tuning/
└── 09_model_monitoring/
```
