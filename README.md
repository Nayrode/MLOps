# CS:GO Match Outcome Prediction — MLOps Pipeline

End-to-end MLOps pipeline for predicting the winner of professional CS:GO matches. Covers feature engineering, model training, evaluation, deployment, and production monitoring.

## Pipeline

```
01_feature_engineering → 02_preprocessing → 08_hyperparameter_tuning (optional)
                                                        ↓
                                         03_train_model_kubernetes
                                                        ↓
                                              04_evaluate_model
                                                        ↓
                                               05_upload_model
                                                        ↓
                                        06_deploy_inference_service
                                                        ↓
                                        07_test_inference_service
                                                        ↓
                                             09_model_monitoring
```

| Step | Folder | What it does |
|------|--------|--------------|
| 01 | `01_feature_engineering` | Build Elo, win rate, H2H, streak, map win rate features |
| 02 | `02_preprocessing` | Chronological train/test split |
| 03 | `03_train_model_kubernetes` | Train Pipeline (scaler + encoder + classifier) as a Kubernetes Job |
| 04 | `04_evaluate_model` | Evaluate on held-out test set, log metrics to MLflow |
| 05 | `05_upload_model` | Register model in MLflow Model Registry |
| 06 | `06_deploy_inference_service` | Deploy as KServe `InferenceService` on Kubernetes |
| 07 | `07_test_inference_service` | Smoke-test the live REST endpoint |
| 08 | `08_hyperparameter_tuning` | Optuna search with `TimeSeriesSplit` CV, logged to MLflow |
| 09 | `09_model_monitoring` | Drift detection and rolling performance tracking |

---

## Quick inference

```bash
uv run python predict.py --team1 NaVi --team2 Astralis --map Dust2
```

---

## Run manually (step by step)

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — `pip install uv`
- Docker + kubectl (steps 03+ in Kubernetes mode)

---

### Step 01 — Feature engineering

```bash
cd 01_feature_engineering
uv sync && uv run python feature_engineering.py
```

Output: `data/df_featured.csv`

---

### Step 02 — Preprocessing

```bash
cd 02_preprocessing
uv sync && uv run python preprocessing.py
```

Output: `data/X_train.csv`, `data/X_test.csv`, `data/y_train.csv`, `data/y_test.csv`

---

### Step 08 — Hyperparameter tuning _(optional, run before step 03)_

```bash
cd 08_hyperparameter_tuning
uv sync && uv run python hyperparameter_tuning.py
```

Output: `data/best_params.json`

---

### Step 03 — Train model

**Local mode:**

```bash
cd 03_train_model_kubernetes
uv sync
DATA_DIR=$(pwd)/../data MODEL_DIR=$(pwd)/../data MODEL_TYPE=xgboost uv run python train.py
```

**Kubernetes mode:**

```bash
cd 03_train_model_kubernetes
docker build -t YOUR_REGISTRY/csgo-trainer:latest .
docker push YOUR_REGISTRY/csgo-trainer:latest
kubectl apply -f job.yaml
kubectl logs -f job/csgo-train-job
```

Output: `data/model.pkl`

---

### Step 04 — Evaluate model

```bash
cd 04_evaluate_model
uv sync && uv run python evaluate_model.py
```

Output: `data/evaluation.json`, `data/confusion_matrix.png`

---

### Step 05 — Upload model

```bash
cd 05_upload_model
uv sync && uv run python upload_model.py
```

---

### Step 06 — Deploy inference service

```bash
cd 06_deploy_inference_service
kubectl apply -f inference_service.yaml
kubectl get inferenceservice
```

---

### Step 07 — Test inference service

```bash
cd 07_test_inference_service
cp .env.example .env   # set INFERENCE_URL
uv sync && uv run python test_inference_service.py
```

---

### Step 09 — Model monitoring

```bash
cd 09_model_monitoring
uv sync && uv run python model_monitoring.py
```

---

## Run with DVC (automated)

Run the full pipeline in one command — DVC skips stages whose inputs haven't changed:

```bash
dvc repro
```

Force re-run everything:

```bash
dvc repro --force
```

---

## Dataset

Source: HLTV professional CS:GO match results (2015-11-03 → 2020-03-18). Data is git-tracked.

| File | Rows | Description |
|------|------|-------------|
| `data/results.csv` | 45,773 | Raw match results |
| `data/df_featured.csv` | 45,773 | + 9 engineered features + target |
| `data/X_train.csv` | 36,618 | Train features (up to 2019-05-22) |
| `data/X_test.csv` | 9,155 | Test features (from 2019-05-22) |
| `data/y_train.csv` | 36,618 | Train target |
| `data/y_test.csv` | 9,155 | Test target |

**Target**: `team_1_wins` (1 = team_1 won, 0 = team_2 won). Class balance ≈ 0.54 / 0.46.

---

## Features

| Feature | Description |
|---------|-------------|
| `elo_diff` | Elo rating difference (team_1 − team_2), K=32 |
| `winrate_10_diff` | Win rate over last 10 matches, difference |
| `winrate_30_diff` | Win rate over last 30 matches, difference |
| `experience_diff` | Total matches played difference |
| `rank_diff` | HLTV rank difference (team_1 − team_2) |
| `h2h_winrate` | Head-to-head win rate for team_1 vs team_2 |
| `streak_diff` | Current win/loss streak difference (e.g. +3 = team_1 on 3-win streak) |
| `map_winrate_diff` | Win rate difference specifically on the played map |
| `_map` | Map name — one-hot encoded inside the sklearn Pipeline |

Features are computed strictly from data **prior to each match** to prevent leakage.
Scaling and map encoding happen inside `model.pkl` — the full inference chain is self-contained.

---

## Model

`sklearn.pipeline.Pipeline`:
1. `ColumnTransformer` — `StandardScaler` on numeric cols, `OneHotEncoder` on `_map`
2. `XGBClassifier` (default) or `LogisticRegression` — switch via `MODEL_TYPE` env var

| Metric | XGBoost | LogReg |
|--------|---------|--------|
| Test accuracy | **0.8043** | 0.769 |
| ROC AUC | **0.8915** | 0.851 |

---

## Repository layout

```
.
├── data/                          # datasets + model (git-tracked)
├── params.yaml                    # pipeline hyperparameters
├── dvc.yaml                       # DVC pipeline DAG
├── predict.py                     # CLI inference: --team1 X --team2 Y --map Z
├── serve.py                       # FastAPI REST API (for KServe deployment)
├── 01_feature_engineering/
├── 02_preprocessing/
├── 03_train_model_kubernetes/
│   ├── train.py
│   ├── Dockerfile
│   └── job.yaml
├── 04_evaluate_model/
├── 05_upload_model/
├── 06_deploy_inference_service/
├── 07_test_inference_service/
├── 08_hyperparameter_tuning/
└── 09_model_monitoring/
```
