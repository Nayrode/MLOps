# CS:GO Match Outcome Prediction — MLOps Pipeline

End-to-end MLOps pipeline for predicting the winner of professional CS:GO matches. Covers data versioning, feature engineering, model training on Kubernetes, evaluation, deployment, and production monitoring.

## Pipeline

```
00_dvc_pull → 01_feature_engineering → 02_preprocessing → 08_hyperparameter_tuning
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
| 00 | `00_dvc_pull` | Pull raw match data from S3 via DVC |
| 01 | `01_feature_engineering` | Build Elo, win rate, H2H features from match history |
| 02 | `02_preprocessing` | Encode map, chronological train/test split |
| 03 | `03_train_model_kubernetes` | Train classifier as a Kubernetes Job |
| 04 | `04_evaluate_model` | Evaluate on held-out test set, log metrics to MLflow |
| 05 | `05_upload_model` | Register model in MLflow Model Registry |
| 06 | `06_deploy_inference_service` | Deploy as KServe `InferenceService` on Kubernetes |
| 07 | `07_test_inference_service` | Smoke-test the live REST endpoint |
| 08 | `08_hyperparameter_tuning` | Optuna search with `TimeSeriesSplit` CV, logged to MLflow |
| 09 | `09_model_monitoring` | Drift detection and rolling performance tracking |

## Dataset

Source: HLTV professional CS:GO match results (2015-11-03 → 2020-03-18).  
Stored in S3 and version-tracked with DVC.

| File | Rows | Description |
|------|------|-------------|
| `data/results.csv` | 45,773 | Raw match results |
| `data/df_featured.csv` | 45,773 | + 6 engineered features |
| `data/X_train.csv` | 36,618 | Train features (up to 2019-05-22) |
| `data/X_test.csv` | 9,155 | Test features (from 2019-05-22) |
| `data/y_train.csv` | 36,618 | Train target |
| `data/y_test.csv` | 9,155 | Test target |

**Raw columns**: `date`, `team_1`, `team_2`, `_map`, `result_1`, `result_2`, `map_winner`, `starting_ct`, `rank_1`, `rank_2`, `match_winner`, …

**Target**: `team_1_wins` (1 = team\_1 won the match, 0 = team\_2 won). Class balance ≈ 0.54 / 0.46.

## Features

| Feature | Description |
|---------|-------------|
| `elo_diff` | Elo rating difference (team\_1 − team\_2), K=32 |
| `winrate_10_diff` | Win rate over last 10 matches, difference |
| `winrate_30_diff` | Win rate over last 30 matches, difference |
| `experience_diff` | Total matches played difference |
| `rank_diff` | HLTV rank difference (team\_1 − team\_2) |
| `h2h_winrate` | Head-to-head win rate for team\_1 vs team\_2 |
| `map_*` (×10) | One-hot encoded map (Cache, Cobblestone, Default, Dust2, Inferno, Mirage, Nuke, Overpass, Train, Vertigo) |

All numeric features are standardised with `StandardScaler`. Features are computed strictly from data **prior to each match** to prevent leakage.

## Setup

Each step is a self-contained Python project managed with [uv](https://github.com/astral-sh/uv).

```bash
cd <step-folder>
uv sync
uv run jupyter lab
```

### DVC remote (S3)

Create `00_dvc_pull/.env` (never commit it):

```
DVC_S3_ACCESS_KEY=...
DVC_S3_SECRET_KEY=...
```

`dvc_pull.ipynb` uses `DVCFileSystem` to stream `data/raw/results.csv` directly from the versioned GitHub repo (`cap4`) without requiring a local DVC setup.

### MLflow tracking

Steps 04, 05, 08 expect `MLFLOW_TRACKING_URI` to point to a running MLflow server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Inference service

Step 07 expects the endpoint URL:

```bash
export INFERENCE_URL=http://<kserve-ingress>/v1/models/<model-name>:predict
```

## Repository layout

```
.
├── data/                          # DVC-tracked datasets (gitignored)
├── 00_dvc_pull/                   # Pull data from S3
├── 01_feature_engineering/        # Stateful feature computation
├── 02_preprocessing/              # Encoding + train/test split
├── 03_train_model_kubernetes/     # Kubernetes training Job
├── 04_evaluate_model/             # Test-set evaluation + MLflow logging
├── 05_upload_model/               # MLflow Model Registry
├── 06_deploy_inference_service/   # KServe InferenceService
├── 07_test_inference_service/     # Endpoint smoke tests
├── 08_hyperparameter_tunning/     # Optuna HP search
└── 09_model_monitoring/           # Drift detection + Grafana metrics
```
