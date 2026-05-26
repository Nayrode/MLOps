"""Kubeflow Pipelines v2 — CS:GO match predictor.

Shared PVC at /data between all steps. DAG:

  dvc_pull → feature_engineering → preprocessing → train → evaluate ─┬─ upload_model → deploy → test_inference
                                                                       └─ monitoring

Usage:
    uv run python pipeline.py              # compile → pipeline.yaml
    uv run python pipeline.py --run        # compile + submit to KFP
"""
import argparse
from kfp import compiler, dsl
from kfp.dsl import Dataset, HTML, Input, Metrics, Model, Output
from kfp.kubernetes import mount_pvc, add_node_selector

BASE_IMAGE = "python:3.11-slim"


# ── Components ────────────────────────────────────────────────────────────────

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==32.0.0"],
)
def create_pvc(namespace: str = "csgo", storage: str = "2Gi"):
    from kubernetes import client as k8s_client, config as k8s_config
    PVC_NAME = "csgo-data-pvc"

    k8s_config.load_incluster_config()
    core_api = k8s_client.CoreV1Api()

    existing = [p.metadata.name for p in core_api.list_namespaced_persistent_volume_claim(namespace).items]
    if PVC_NAME in existing:
        print(f"PVC {PVC_NAME} already exists, skipping.")
        return

    pvc = k8s_client.V1PersistentVolumeClaim(
        metadata=k8s_client.V1ObjectMeta(name=PVC_NAME, namespace=namespace),
        spec=k8s_client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=k8s_client.V1ResourceRequirements(requests={"storage": storage}),
        ),
    )
    core_api.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
    print(f"Created PVC {PVC_NAME} ({storage}) in namespace {namespace}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3"],
)
def dvc_pull():
    import urllib.request
    import pandas as pd
    DATA = "/data"
    REPO  = "https://raw.githubusercontent.com/Nayrode/MLOps/main"

    urllib.request.urlretrieve(f"{REPO}/data/results.csv", f"{DATA}/results.csv")
    df = pd.read_csv(f"{DATA}/results.csv")
    print(f"Downloaded {len(df)} rows → {DATA}/results.csv")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "numpy==2.2.3"],
)
def feature_engineering():
    import numpy as np
    import pandas as pd
    DATA = "/data"

    def _streak(history):
        if not history:
            return 0
        val = history[-1]
        streak = 0
        for r in reversed(history):
            if r == val: streak += 1
            else: break
        return streak if val == 1 else -streak

    def build_state(df):
        state = {"elo": {}, "matches_played": {}, "win_history": {}, "h2h": {}, "map_history": {}}
        for team in pd.concat([df["team_1"], df["team_2"]]).unique():
            state["elo"][team] = 1500
            state["matches_played"][team] = 0
            state["win_history"][team] = []
            state["map_history"][team] = {}
        return state

    def compute_features(match, state):
        team, opp, map_name = match["team_1"], match["team_2"], match["_map"]
        th  = state["win_history"].get(team, [])
        oh  = state["win_history"].get(opp, [])
        h2h = state["h2h"].get((team, opp), [])
        mht = state["map_history"].get(team, {}).get(map_name, [])
        mho = state["map_history"].get(opp,  {}).get(map_name, [])
        return {
            "elo_diff":         state["elo"].get(team, 1500) - state["elo"].get(opp, 1500),
            "winrate_10_diff":  (np.mean(th[-10:]) if th else 0) - (np.mean(oh[-10:]) if oh else 0),
            "winrate_30_diff":  (np.mean(th[-30:]) if th else 0) - (np.mean(oh[-30:]) if oh else 0),
            "experience_diff":  state["matches_played"].get(team, 0) - state["matches_played"].get(opp, 0),
            "rank_diff":        match["rank_1"] - match["rank_2"],
            "h2h_winrate":      np.mean(h2h) if h2h else 0.5,
            "streak_diff":      _streak(th) - _streak(oh),
            "map_winrate_diff": (np.mean(mht) if mht else 0.5) - (np.mean(mho) if mho else 0.5),
        }

    def update_state(match, state, k=32):
        team, opp, map_name = match["team_1"], match["team_2"], match["_map"]
        result = int(match["match_winner"] == 1)
        r_t, r_o = state["elo"].get(team, 1500), state["elo"].get(opp, 1500)
        exp = 1 / (1 + 10 ** ((r_o - r_t) / 400))
        state["elo"][team] = r_t + k * (result - exp)
        state["elo"][opp]  = r_o + k * ((1 - result) - (1 - exp))
        state["win_history"].setdefault(team, []).append(result)
        state["win_history"].setdefault(opp,  []).append(1 - result)
        state["matches_played"][team] = state["matches_played"].get(team, 0) + 1
        state["matches_played"][opp]  = state["matches_played"].get(opp,  0) + 1
        state["h2h"].setdefault((team, opp), []).append(result)
        state["map_history"].setdefault(team, {}).setdefault(map_name, []).append(result)
        state["map_history"].setdefault(opp,  {}).setdefault(map_name, []).append(1 - result)
        return state

    df = pd.read_csv(f"{DATA}/results.csv", index_col=0, parse_dates=["date"])
    print(f"Loaded {len(df)} rows")
    state = build_state(df)
    X, y = [], []
    for _, match in df.iterrows():
        X.append(compute_features(match, state))
        y.append(int(match["match_winner"] == 1))
        state = update_state(match, state)
    out = pd.concat([df, pd.DataFrame(X), pd.Series(y, name="team_1_wins")], axis=1)
    out.to_csv(f"{DATA}/df_featured.csv")
    print(f"Saved {len(out)} rows → {DATA}/df_featured.csv")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3"],
)
def preprocessing(train_ratio: float = 0.8):
    import pandas as pd
    DATA = "/data"

    FEATURE_COLS = ["elo_diff", "winrate_10_diff", "winrate_30_diff",
                    "experience_diff", "rank_diff", "h2h_winrate",
                    "streak_diff", "map_winrate_diff", "_map"]

    df = pd.read_csv(f"{DATA}/df_featured.csv", index_col=0, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    train_df[FEATURE_COLS].to_csv(f"{DATA}/X_train.csv")
    train_df["team_1_wins"].to_csv(f"{DATA}/y_train.csv")
    test_df[FEATURE_COLS].to_csv(f"{DATA}/X_test.csv")
    test_df["team_1_wins"].to_csv(f"{DATA}/y_test.csv")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1", "xgboost==3.2.0", "joblib==1.5.0"],
)
def train(model_type: str = "xgboost"):
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier
    DATA = "/data"

    NUMERIC_COLS = ["elo_diff", "winrate_10_diff", "winrate_30_diff",
                    "experience_diff", "rank_diff", "h2h_winrate",
                    "streak_diff", "map_winrate_diff"]

    X = pd.read_csv(f"{DATA}/X_train.csv", index_col=0)
    y = pd.read_csv(f"{DATA}/y_train.csv", index_col=0).squeeze()

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("scaler",  StandardScaler(), NUMERIC_COLS),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
        ])),
        ("classifier", XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, eval_metric="logloss", random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    print(f"Train accuracy: {pipeline.score(X, y):.4f}")
    joblib.dump(pipeline, f"{DATA}/model.pkl")
    print(f"Saved model → {DATA}/model.pkl")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1", "xgboost==3.2.0", "joblib==1.5.0"],
)
def evaluate(eval_metrics: Output[Metrics]):
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    DATA = "/data"

    pipeline = joblib.load(f"{DATA}/model.pkl")
    X = pd.read_csv(f"{DATA}/X_test.csv", index_col=0)
    y = pd.read_csv(f"{DATA}/y_test.csv", index_col=0).squeeze()

    acc = accuracy_score(y, pipeline.predict(X))
    auc = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    eval_metrics.log_metric("accuracy", acc)
    eval_metrics.log_metric("roc_auc", auc)

    with open(f"{DATA}/evaluation.json", "w") as f:
        json.dump({"accuracy": acc, "roc_auc": auc}, f, indent=2)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["mlflow==2.19.0", "scikit-learn==1.6.1", "xgboost==3.2.0",
                         "joblib==1.5.0", "boto3==1.36.0"],
)
def upload_model(accuracy_threshold: float = 0.60,
                 mlflow_tracking_uri: str = "http://mlflow.kubeflow.svc.cluster.local:5000"):
    import json
    import joblib
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    DATA = "/data"
    MODEL_NAME = "csgo-match-predictor"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MODEL_NAME)

    model = joblib.load(f"{DATA}/model.pkl")
    with open(f"{DATA}/evaluation.json") as f:
        metrics = json.load(f)

    print(f"Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    if metrics["accuracy"] < accuracy_threshold:
        raise ValueError(f"Accuracy {metrics['accuracy']:.4f} below threshold {accuracy_threshold}")

    with mlflow.start_run(run_name="kubeflow-pipeline"):
        mlflow.log_metrics(metrics)
        mlflow.log_params({"model_type": type(model.named_steps["classifier"]).__name__})
        info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        model_uri = info.model_uri

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    latest = versions[0]
    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest.version, stage="Staging",
    )
    print(f"Version {latest.version} → Staging  (uri: {model_uri})")

    with open(f"{DATA}/model_uri.txt", "w") as f:
        f.write(model_uri)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==32.0.0"],
)
def deploy(namespace: str = "csgo"):
    import time
    from kubernetes import client as k8s_client, config as k8s_config
    DATA = "/data"
    MODEL_NAME = "csgo-match-predictor"

    with open(f"{DATA}/model_uri.txt") as f:
        model_uri = f.read().strip()

    k8s_config.load_incluster_config()
    custom_api = k8s_client.CustomObjectsApi()

    body = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {"name": MODEL_NAME, "namespace": namespace},
        "spec": {
            "predictor": {
                "sklearn": {
                    "storageUri": model_uri,
                    "protocolVersion": "v2",
                }
            }
        },
    }

    try:
        custom_api.create_namespaced_custom_object(
            group="serving.kserve.io", version="v1beta1",
            namespace=namespace, plural="inferenceservices", body=body,
        )
        print(f"Created InferenceService {MODEL_NAME}")
    except k8s_client.exceptions.ApiException as e:
        if e.status == 409:
            custom_api.replace_namespaced_custom_object(
                group="serving.kserve.io", version="v1beta1",
                namespace=namespace, plural="inferenceservices",
                name=MODEL_NAME, body=body,
            )
            print(f"Updated InferenceService {MODEL_NAME}")
        else:
            raise

    for _ in range(60):
        isvc = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io", version="v1beta1",
            namespace=namespace, plural="inferenceservices", name=MODEL_NAME,
        )
        conditions = isvc.get("status", {}).get("conditions", [])
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions):
            url = isvc["status"]["url"]
            print(f"InferenceService ready at {url}")
            with open(f"{DATA}/inference_url.txt", "w") as f:
                f.write(url)
            return
        time.sleep(5)
    raise TimeoutError(f"InferenceService {MODEL_NAME} not ready after 300s")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["requests==2.32.3", "pandas==2.2.3"],
)
def test_inference(n_samples: int = 10):
    import requests
    import pandas as pd
    DATA = "/data"
    MODEL_NAME = "csgo-match-predictor"

    with open(f"{DATA}/inference_url.txt") as f:
        base_url = f.read().strip()

    X_test = pd.read_csv(f"{DATA}/X_test.csv", index_col=0)
    y_test = pd.read_csv(f"{DATA}/y_test.csv", index_col=0).squeeze()

    sample   = X_test.sample(n_samples, random_state=42)
    expected = y_test.loc[sample.index].tolist()

    url      = f"{base_url}/v1/models/{MODEL_NAME}:predict"
    response = requests.post(url, json={"instances": sample.values.tolist()}, timeout=30)
    response.raise_for_status()

    predictions = response.json()["predictions"]
    correct = sum(p == e for p, e in zip(predictions, expected))
    print(f"Smoke test: {correct}/{n_samples} correct")
    assert len(predictions) == n_samples, f"Expected {n_samples}, got {len(predictions)}"
    assert all(p in [0, 1] for p in predictions), "Predictions must be 0 or 1"
    print("Smoke test passed.")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1", "xgboost==3.2.0",
                         "joblib==1.5.0", "evidently==0.7.21"],
)
def monitoring(drift_report: Output[HTML], monitoring_metrics: Output[Metrics],
               drift_threshold: float = 0.20):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    from evidently import Report
    from evidently.presets import DataDriftPreset
    DATA = "/data"

    reference = pd.read_csv(f"{DATA}/X_train.csv", index_col=0)
    current   = pd.read_csv(f"{DATA}/X_test.csv",  index_col=0)
    y_test    = pd.read_csv(f"{DATA}/y_test.csv",  index_col=0).squeeze()

    print(f"Reference (train) : {len(reference)} rows")
    print(f"Current   (test)  : {len(current)} rows")

    snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
    snapshot.save_html(drift_report.path)

    metrics      = snapshot.dict()["metrics"]
    drift_result = metrics[0]["value"]
    share_drifted = drift_result["share"]
    n_drifted     = int(drift_result["count"])
    n_total       = len(reference.columns)

    print(f"Drifted columns: {n_drifted}/{n_total} ({share_drifted:.1%})")
    if share_drifted > drift_threshold:
        print(f"ALERT: drift {share_drifted:.1%} exceeds threshold {drift_threshold:.1%}")
    else:
        print("No significant drift detected.")

    model  = joblib.load(f"{DATA}/model.pkl")
    y_pred = model.predict(current)
    y_prob = model.predict_proba(current)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nPerformance on current data:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC AUC  : {auc:.4f}")

    monitoring_metrics.log_metric("share_drifted", share_drifted)
    monitoring_metrics.log_metric("n_drifted_columns", float(n_drifted))
    monitoring_metrics.log_metric("accuracy", acc)
    monitoring_metrics.log_metric("roc_auc", auc)


# ── Pipeline DAG ──────────────────────────────────────────────────────────────

@dsl.pipeline(name="csgo-match-predictor")
def csgo_pipeline(
    model_type: str = "xgboost",
    train_ratio: float = 0.8,
    accuracy_threshold: float = 0.60,
    drift_threshold: float = 0.20,
    mlflow_tracking_uri: str = "http://mlflow.kubeflow.svc.cluster.local:5000",
    deploy_namespace: str = "csgo",
):
    pvc  = create_pvc(namespace=deploy_namespace, storage="2Gi")
    pull = dvc_pull().after(pvc)
    fe   = feature_engineering().after(pull)
    pre  = preprocessing(train_ratio=train_ratio).after(fe)
    tr   = train(model_type=model_type).after(pre)
    ev   = evaluate().after(tr)  # type: ignore[call-arg]
    up   = upload_model(
        accuracy_threshold=accuracy_threshold,
        mlflow_tracking_uri=mlflow_tracking_uri,
    ).after(ev)
    dep  = deploy(namespace=deploy_namespace).after(up)
    test = test_inference().after(dep)
    mon  = monitoring(drift_threshold=drift_threshold).after(ev)  # type: ignore[call-arg]

    for task in [pull, fe, pre, tr, ev, up, dep, test, mon]:
        mount_pvc(task, pvc_name="csgo-data-pvc", mount_path="/data")

    # Force tous les pods sur les workers (les CP n'ont pas nfs-common)
    for task in [pvc, pull, fe, pre, tr, ev, up, dep, test, mon]:
        add_node_selector(task, label_key="node-role.kubernetes.io/worker", label_value="worker")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  action="store_true")
    parser.add_argument("--host", default="http://localhost:3000")
    args = parser.parse_args()

    compiler.Compiler().compile(csgo_pipeline, "pipeline.yaml")
    print("Compiled → pipeline.yaml")

    if args.run:
        import kfp
        client = kfp.Client(host=args.host)
        run = client.create_run_from_pipeline_func(
            csgo_pipeline,
            arguments={"model_type": "xgboost"},
            run_name="csgo-run",
        )
        print(f"Submitted → {run.run_id}")
