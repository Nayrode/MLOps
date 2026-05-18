# 08 — Hyperparameter Tuning

Searches for the best model hyperparameters using cross-validated optimization over the training set.

## Inputs

- `data/X_train.csv`
- `data/y_train.csv`

## What it does

- Runs a search (e.g. Optuna or `GridSearchCV`) over the model's hyperparameter space
- Uses time-series-aware cross-validation (`TimeSeriesSplit`) to respect chronological order
- Logs each trial to MLflow

## Outputs

- Best hyperparameters logged as an MLflow run
- Optionally retriggers step 03 with the best config

## Run

```bash
uv sync
uv run jupyter lab
```

Open `hyperparameter_tuning.ipynb` and run all cells.

## Notes

- Run this before step 03 to find good hyperparameters, then train a final model with them
