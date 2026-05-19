# 08 — Hyperparameter Tuning

Searches for the best classifier hyperparameters using Optuna with time-series-aware cross-validation.

## Inputs

- `data/X_train.csv`
- `data/y_train.csv`

## Search space

**XGBoost** (default):

| Param | Range |
|-------|-------|
| `n_estimators` | 100 – 500 |
| `learning_rate` | log-uniform [1e-3, 0.3] |
| `max_depth` | 3 – 10 |

**LogisticRegression** (`MODEL_TYPE=logreg`):

| Param | Range |
|-------|-------|
| `C` | log-uniform [1e-3, 100] |
| `solver` | `lbfgs`, `liblinear` |
| `max_iter` | 200 – 2000 (step 200) |

CV strategy: `TimeSeriesSplit(n_splits=5)` — respects chronological order.
Objective: maximise mean ROC AUC across folds.

## Outputs

- `data/best_params.json` — `{"model_type": "xgboost", "params": {...}, "cv_roc_auc": ...}`

Step 03 reads `best_params.json` automatically if `model_type` matches `MODEL_TYPE`.

## Environment

```
MLFLOW_TRACKING_URI=http://localhost:5000   # optional
MODEL_TYPE=xgboost                          # or logreg
```

## Run

```bash
cp .env.example .env
uv sync && uv run python hyperparameter_tuning.py
```

Then re-run step 03 to train with the optimised parameters.
