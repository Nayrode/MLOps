# 08 — Hyperparameter Tuning

Searches for the best `LogisticRegression` hyperparameters using Optuna with time-series-aware cross-validation.

## Inputs

- `data/X_train.csv`
- `data/y_train.csv`

## Search space

| Param | Range |
|-------|-------|
| `C` | log-uniform [1e-3, 100] |
| `solver` | `lbfgs`, `liblinear` |
| `max_iter` | 200 – 2000 (step 200) |

CV strategy: `TimeSeriesSplit(n_splits=5)` — respects chronological order, no future leakage.  
Objective: maximise mean ROC AUC across folds.  
Each trial is logged as a nested MLflow run under `csgo-hp-tuning`.

## Outputs

- `data/best_params.json` — `{"params": {...}, "cv_roc_auc": ...}`
- MLflow experiment `csgo-hp-tuning` with all trial runs

Step 03 automatically reads `best_params.json` if it exists.

## Environment

Create `.env` (optional — defaults to local `mlruns/`):

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Run

```bash
uv sync
uv run jupyter lab
```

Open `hyperparameter_tuning.ipynb` and run all cells. Then re-run step 03.
