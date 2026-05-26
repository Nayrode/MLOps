"""
Tests rapides pour les composants KFP sans passer par le cluster.

Usage:
    uv run python test_pipeline.py                  # tous les tests locaux
    uv run python test_pipeline.py --step train     # un seul step
    uv run python test_pipeline.py --kfp-local      # KFP SubprocessRunner (installe les deps)
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

class Artifact:
    def __init__(self, path):
        self.path = str(path)

class Metrics:
    def log_metric(self, key, value):
        print(f"  metric {key} = {value}")


def _make_test_data(tmp: Path):
    """Génère un dataset CS:GO minimal pour les tests."""
    maps = ["de_dust2", "de_inferno", "de_mirage"]
    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date":         pd.date_range("2020-01-01", periods=n, freq="D"),
        "team_1":       rng.choice(["TeamA", "TeamB", "TeamC", "TeamD"], n),
        "team_2":       rng.choice(["TeamE", "TeamF", "TeamG", "TeamH"], n),
        "rank_1":       rng.integers(1, 20, n),
        "rank_2":       rng.integers(1, 20, n),
        "_map":         rng.choice(maps, n),
        "match_winner": rng.integers(1, 3, n),
    })
    raw = tmp / "raw.csv"
    df.to_csv(raw)
    return raw


# ── Tests par step ────────────────────────────────────────────────────────────

def test_dvc_pull(tmp: Path):
    print("▶ dvc_pull")
    from pipeline import dvc_pull
    out = Artifact(tmp / "raw.csv")
    dvc_pull.python_func(raw_data=out)
    df = pd.read_csv(out.path)
    assert len(df) > 0, "dvc_pull: dataset vide"
    print(f"  OK — {len(df)} lignes")
    return out


def test_feature_engineering(tmp: Path, raw: Artifact):
    print("▶ feature_engineering")
    from pipeline import feature_engineering
    out = Artifact(tmp / "featured.csv")
    feature_engineering.python_func(raw_data=raw, featured_data=out)
    df = pd.read_csv(out.path)
    assert "elo_diff" in df.columns, "feature_engineering: colonne elo_diff manquante"
    print(f"  OK — {len(df)} lignes, {len(df.columns)} colonnes")
    return out


def test_preprocessing(tmp: Path, featured: Artifact):
    print("▶ preprocessing")
    from pipeline import preprocessing
    X_train = Artifact(tmp / "X_train.csv")
    y_train = Artifact(tmp / "y_train.csv")
    X_test  = Artifact(tmp / "X_test.csv")
    y_test  = Artifact(tmp / "y_test.csv")
    preprocessing.python_func(
        featured_data=featured,
        train_ratio=0.8,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
    )
    n_train = len(pd.read_csv(X_train.path))
    n_test  = len(pd.read_csv(X_test.path))
    print(f"  OK — train={n_train} test={n_test}")
    return X_train, y_train, X_test, y_test


def test_train_local(tmp: Path, X_train: Artifact, y_train: Artifact):
    """Teste le corps de train_func directement (sans TrainingClient/cluster)."""
    print("▶ train (local — sans TrainingClient)")
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    NUMERIC_COLS = ["elo_diff", "winrate_10_diff", "winrate_30_diff",
                    "experience_diff", "rank_diff", "h2h_winrate",
                    "streak_diff", "map_winrate_diff"]

    X = pd.read_csv(X_train.path, index_col=0)
    y = pd.read_csv(y_train.path, index_col=0).squeeze()

    model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("scaler",  StandardScaler(), NUMERIC_COLS),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
        ])),
        ("classifier", XGBClassifier(n_estimators=10, random_state=42)),  # 10 arbres pour aller vite
    ])
    model.fit(X, y)
    acc = model.score(X, y)
    model_path = tmp / "model.pkl"
    joblib.dump(model, model_path)
    print(f"  OK — accuracy={acc:.4f}, model → {model_path}")
    return Artifact(model_path)


def test_evaluate(tmp: Path, model: Artifact, X_test: Artifact, y_test: Artifact):
    print("▶ evaluate")
    from pipeline import evaluate
    metrics     = Metrics()
    eval_results = Artifact(tmp / "eval.json")
    evaluate.python_func(
        model=model, X_test=X_test, y_test=y_test,
        metrics=metrics, eval_results=eval_results,
    )
    print("  OK")
    return eval_results


def test_monitoring(tmp: Path, X_train: Artifact, X_test: Artifact,
                    y_test: Artifact, model: Artifact):
    print("▶ monitoring")
    from pipeline import monitoring
    drift_report       = Artifact(tmp / "drift.html")
    monitoring_metrics = Metrics()
    monitoring.python_func(
        X_train=X_train, X_test=X_test, y_test=y_test, model=model,
        drift_threshold=0.20,
        drift_report=drift_report,
        monitoring_metrics=monitoring_metrics,
    )
    print("  OK")


# ── KFP Local Runner ──────────────────────────────────────────────────────────

def run_kfp_local():
    """Lance la pipeline complète via kfp.local.SubprocessRunner (sans cluster)."""
    import kfp.local
    from pipeline import csgo_pipeline

    kfp.local.init(runner=kfp.local.SubprocessRunner(use_venv=False))
    csgo_pipeline(
        model_type="xgboost",
        train_ratio=0.8,
        accuracy_threshold=0.50,  # seuil bas pour ne pas bloquer
        drift_threshold=0.99,
        mlflow_tracking_uri="http://localhost:5000",
        deploy_namespace="csgo",
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["dvc_pull", "feature_engineering",
                                           "preprocessing", "train", "evaluate",
                                           "monitoring", "all"],
                        default="all")
    parser.add_argument("--kfp-local", action="store_true",
                        help="Lance via kfp.local.SubprocessRunner")
    args = parser.parse_args()

    if args.kfp_local:
        run_kfp_local()
    else:
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            _make_test_data(tmp)  # données fake dans tmp

            step = args.step

            raw      = test_dvc_pull(tmp)                                    if step in ("dvc_pull",            "all") else Artifact(tmp/"raw.csv")
            featured = test_feature_engineering(tmp, raw)                    if step in ("feature_engineering", "all") else Artifact(tmp/"featured.csv")
            splits   = test_preprocessing(tmp, featured)                     if step in ("preprocessing",       "all") else None
            X_train, y_train, X_test, y_test = splits or (None,)*4
            model    = test_train_local(tmp, X_train, y_train)               if step in ("train",               "all") else None
            _        = test_evaluate(tmp, model, X_test, y_test)             if step in ("evaluate",            "all") else None
            _        = test_monitoring(tmp, X_train, X_test, y_test, model)  if step in ("monitoring",          "all") else None

        print("\n✓ Tous les tests passent.")
