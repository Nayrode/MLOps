"""Kubeflow Pipelines v2 — CS:GO match predictor.

DAG:
  dvc_pull → feature_engineering → preprocessing → train → evaluate ─┬─ upload_model → deploy → test_inference
                                                                       └─ monitoring

Usage:
    uv run python pipeline.py              # compile → pipeline.yaml
    uv run python pipeline.py --run        # compile + submit to KFP
"""
import argparse
from kfp import compiler, dsl
from kfp.dsl import Dataset, Model, Metrics, Artifact, Input, Output
from kfp.kubernetes import add_node_selector, mount_pvc, use_secret_as_env

BASE_IMAGE = "python:3.11-slim"


# ── Components ────────────────────────────────────────────────────────────────

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3"],
)
def dvc_pull(raw_data: Output[Dataset]):
    import urllib.request
    import pandas as pd

    REPO = "https://raw.githubusercontent.com/Nayrode/MLOps/main"
    urllib.request.urlretrieve(f"{REPO}/data/results.csv", raw_data.path)
    df = pd.read_csv(raw_data.path)
    print(f"Downloaded {len(df)} rows → {raw_data.path}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "numpy==2.4.6"],
)
def feature_engineering(
    raw_data: Input[Dataset],
    featured_data: Output[Dataset],
):
    import numpy as np
    import pandas as pd

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

    df = pd.read_csv(raw_data.path, index_col=0, parse_dates=["date"])
    print(f"Loaded {len(df)} rows")
    state = build_state(df)
    X, y = [], []
    for _, match in df.iterrows():
        X.append(compute_features(match, state))
        y.append(int(match["match_winner"] == 1))
        state = update_state(match, state)
    out = pd.concat([df, pd.DataFrame(X), pd.Series(y, name="team_1_wins")], axis=1)
    out.to_csv(featured_data.path)
    print(f"Saved {len(out)} rows → {featured_data.path}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3"],
)
def preprocessing(
    featured_data: Input[Dataset],
    train_ratio: float,
    X_train: Output[Dataset],
    y_train: Output[Dataset],
    X_test: Output[Dataset],
    y_test: Output[Dataset],
):
    import pandas as pd

    FEATURE_COLS = ["elo_diff", "winrate_10_diff", "winrate_30_diff",
                    "experience_diff", "rank_diff", "h2h_winrate",
                    "streak_diff", "map_winrate_diff", "_map"]

    df = pd.read_csv(featured_data.path, index_col=0, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    train_df[FEATURE_COLS].to_csv(X_train.path)
    train_df["team_1_wins"].to_csv(y_train.path)
    test_df[FEATURE_COLS].to_csv(X_test.path)
    test_df["team_1_wins"].to_csv(y_test.path)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")



@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==30.1.0"],
)
def create_train_pvc(namespace: str = "csgo", pvc_name: str = "csgo-train-data"):
    from kubernetes import client as k8s, config as k8s_config
    k8s_config.load_incluster_config()
    core = k8s.CoreV1Api()

    existing = [p.metadata.name for p in core.list_namespaced_persistent_volume_claim(namespace).items]
    if pvc_name in existing:
        print(f"PVC {pvc_name} already exists, skipping.")
        return

    pvc = k8s.V1PersistentVolumeClaim(
        metadata=k8s.V1ObjectMeta(name=pvc_name, namespace=namespace),
        spec=k8s.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteMany"],
            resources=k8s.V1ResourceRequirements(requests={"storage": "1Gi"}),
        ),
    )
    core.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
    print(f"Created RWX PVC {pvc_name} in namespace {namespace}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==30.1.0", "boto3==1.43.14", "joblib==1.5.3"],
)
def train(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    model_type: str,
    model: Output[Model],
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    namespace: str = "csgo",
    s3_endpoint: str = "http://seaweedfs.kubeflow.svc.cluster.local:8333",
    s3_bucket: str = "mlpipeline",
):
    import os
    import time
    import boto3
    from kubernetes import client as k8s, config as k8s_config

    s3_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    s3_secret  = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

    # KFP artifact URIs use minio:// scheme — convert to s3:// for boto3.
    x_uri = X_train.uri.replace("minio://", "s3://")
    y_uri = y_train.uri.replace("minio://", "s3://")

    ts       = int(time.time())
    job_name = f"csgo-train-{ts}"

    # Upload directement vers model.uri (private-artifacts/) — les creds ont accès en écriture là.
    # minio:// → s3:// pour boto3
    model_uri = model.uri.replace("minio://", "s3://")
    model_bucket, model_s3_key = model_uri.replace("s3://", "").split("/", 1)

    training_script = """\
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "pandas==2.2.3", "scikit-learn==1.8.0", "xgboost==3.2.0",
    "joblib==1.5.3", "boto3==1.43.14"])
import io, os, tempfile, boto3, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

s3 = boto3.client("s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

def download_df(uri):
    bucket, key = uri.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)

X = download_df(os.environ["X_URI"])
y = download_df(os.environ["Y_URI"]).squeeze()
print(f"Loaded X_train {X.shape}, y_train {y.shape}")

NUMERIC_COLS = ["elo_diff","winrate_10_diff","winrate_30_diff",
    "experience_diff","rank_diff","h2h_winrate","streak_diff","map_winrate_diff"]

pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("scaler",  StandardScaler(), NUMERIC_COLS),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
    ])),
    ("classifier", XGBClassifier(
        n_estimators=int(os.environ["N_ESTIMATORS"]),
        learning_rate=float(os.environ["LEARNING_RATE"]),
        max_depth=int(os.environ["MAX_DEPTH"]),
        eval_metric="logloss", random_state=42, device="cuda",
    )),
])
pipeline.fit(X, y)
print(f"Train accuracy: {pipeline.score(X, y):.4f}")

with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    joblib.dump(pipeline, f.name)
    s3.upload_file(f.name, os.environ["MODEL_BUCKET"], os.environ["MODEL_KEY"])
print(f"Model saved → s3://{os.environ['MODEL_BUCKET']}/{os.environ['MODEL_KEY']}")
"""

    k8s_config.load_incluster_config()
    custom_api = k8s.CustomObjectsApi()

    gpu_patch = {
        "manager": "csgo.io/train",
        "trainingRuntimeSpec": {
            "template": {
                "spec": {
                    "replicatedJobs": [{
                        "name": "node",
                        "template": {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "tolerations": [{
                                            "key": "nvidia.com/gpu",
                                            "operator": "Equal",
                                            "value": "present",
                                            "effect": "NoSchedule",
                                        }],
                                        "nodeSelector": {
                                            "node-role.kubernetes.io/gpu": "true",
                                        },
                                    }
                                }
                            }
                        }
                    }]
                }
            }
        },
    }

    body = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "runtimeRef": {
                "apiGroup": "trainer.kubeflow.org",
                "kind": "ClusterTrainingRuntime",
                "name": "xgboost-distributed",
            },
            "runtimePatches": [gpu_patch],
            "trainer": {
                "image": "python:3.11-slim",
                "command": ["python", "-c", training_script],
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {"nvidia.com/gpu": "1"},
                    "limits":   {"nvidia.com/gpu": "1"},
                },
                "env": [
                    {"name": "S3_ENDPOINT",           "value": s3_endpoint},
                    {"name": "S3_BUCKET",             "value": s3_bucket},
                    {"name": "AWS_ACCESS_KEY_ID",     "value": s3_key_id},
                    {"name": "AWS_SECRET_ACCESS_KEY", "value": s3_secret},
                    {"name": "X_URI",                 "value": x_uri},
                    {"name": "Y_URI",                 "value": y_uri},
                    {"name": "MODEL_BUCKET",          "value": model_bucket},
                    {"name": "MODEL_KEY",             "value": model_s3_key},
                    {"name": "N_ESTIMATORS",          "value": str(n_estimators)},
                    {"name": "LEARNING_RATE",         "value": str(learning_rate)},
                    {"name": "MAX_DEPTH",             "value": str(max_depth)},
                ],
            },
        },
    }

    custom_api.create_namespaced_custom_object(
        group="trainer.kubeflow.org", version="v1alpha1",
        namespace=namespace, plural="trainjobs", body=body,
    )
    print(f"TrainJob '{job_name}' soumis dans '{namespace}'.")

    # Poll until succeeded or failed
    interval, elapsed, timeout = 15, 0, 900
    while elapsed < timeout:
        job = custom_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", name=job_name,
        )
        for js in job.get("status", {}).get("jobsStatus", []):
            if js.get("name") == "node":
                active, succeeded, failed = js.get("active", 0), js.get("succeeded", 0), js.get("failed", 0)
                print(f"  [{elapsed:3d}s] active={active} succeeded={succeeded} failed={failed}")
                if succeeded >= 1:
                    print("TrainJob terminé avec succès.")
                    break
                if failed >= 1:
                    raise RuntimeError(f"TrainJob '{job_name}' a échoué.")
        else:
            time.sleep(interval)
            elapsed += interval
            continue
        break
    else:
        raise TimeoutError(f"TrainJob '{job_name}' timeout après {timeout}s.")

    # Le TrainJob a uploadé vers model.uri via S3, mais le FUSE local ne le voit pas.
    # On re-télécharge vers model.path pour que KFP trouve le fichier.
    s3 = boto3.client("s3", endpoint_url=s3_endpoint,
                      aws_access_key_id=s3_key_id, aws_secret_access_key=s3_secret)
    s3.download_file(model_bucket, model_s3_key, model.path)
    print(f"Model artifact → {model.path}")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.8.0", "xgboost==3.2.0", "joblib==1.5.3"],
)
def evaluate(
    model: Input[Model],
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    metrics: Output[Metrics],
    eval_results: Output[Artifact],
):
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score

    pipeline = joblib.load(model.path)
    X = pd.read_csv(X_test.path, index_col=0)
    y = pd.read_csv(y_test.path, index_col=0).squeeze()

    acc = accuracy_score(y, pipeline.predict(X))
    auc = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    metrics.log_metric("accuracy", acc)
    metrics.log_metric("roc_auc",  auc)

    with open(eval_results.path, "w") as f:
        json.dump({"accuracy": acc, "roc_auc": auc}, f, indent=2)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3==1.43.14", "requests==2.32.3"],
)
def upload_model(
    model: Input[Model],
    eval_results: Input[Artifact],
    accuracy_threshold: float,
    s3_endpoint: str,
    model_registry_url: str,
    model_uri: Output[Artifact],
):
    import json
    import os
    import uuid
    import boto3
    import requests as http

    MODEL_NAME = "csgo-match-predictor"
    BUCKET = "mlpipeline"
    MR_API = f"{model_registry_url}/api/model_registry/v1alpha3"

    with open(eval_results.path) as f:
        metrics_data = json.load(f)

    print(f"Accuracy: {metrics_data['accuracy']:.4f} | AUC: {metrics_data['roc_auc']:.4f}")
    if metrics_data["accuracy"] < accuracy_threshold:
        raise ValueError(f"Accuracy {metrics_data['accuracy']:.4f} below threshold {accuracy_threshold}")

    version_name = uuid.uuid4().hex[:8]
    s3_key = f"private-artifacts/csgo/models/{MODEL_NAME}/{version_name}/model.joblib"
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3.upload_file(model.path, BUCKET, s3_key)
    artifact_uri = f"s3://{BUCKET}/{s3_key}"
    print(f"Uploaded model → {artifact_uri}")

    # Create or retrieve RegisteredModel
    resp = http.post(
        f"{MR_API}/registered_models",
        json={"name": MODEL_NAME, "description": "CS:GO match predictor"},
        timeout=30,
    )
    print(f"POST registered_models → {resp.status_code}: {resp.text}")
    if resp.status_code == 409:
        resp = http.get(f"{MR_API}/registered_models", timeout=30)
        resp.raise_for_status()
        print(f"GET registered_models → {resp.status_code}: {resp.text}")
        rm_id = next(rm["id"] for rm in resp.json()["items"] if rm["name"] == MODEL_NAME)
    else:
        resp.raise_for_status()
        rm_id = resp.json()["id"]
    print(f"RegisteredModel id={rm_id}")

    # Create ModelVersion with evaluation metrics as custom properties
    mv_resp = http.post(
        f"{MR_API}/model_versions",
        json={
            "name": version_name,
            "registeredModelId": rm_id,
            "customProperties": {
                "accuracy": {"metadataType": "MetadataDoubleValue", "double_value": metrics_data["accuracy"]},
                "roc_auc":  {"metadataType": "MetadataDoubleValue", "double_value": metrics_data["roc_auc"]},
            },
        },
        timeout=30,
    )
    print(f"POST model_versions → {mv_resp.status_code}: {mv_resp.text}")
    mv_resp.raise_for_status()
    mv_id = mv_resp.json()["id"]
    print(f"ModelVersion id={mv_id}")

    # Link artifact URI to the model version
    ma_resp = http.post(
        f"{MR_API}/model_versions/{mv_id}/artifacts",
        json={
            "name": f"{MODEL_NAME}-{version_name}",
            "uri": artifact_uri,
            "artifactType": "model-artifact",
            "state": "LIVE",
        },
        timeout=30,
    )
    print(f"POST model_versions/{mv_id}/artifacts → {ma_resp.status_code}: {ma_resp.text}")
    ma_resp.raise_for_status()

    print(f"Registered '{MODEL_NAME}' version '{version_name}' (id={mv_id}) → {artifact_uri}")

    with open(model_uri.path, "w") as f:
        f.write(artifact_uri)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==36.0.0"],
)
def deploy(
    model_uri: Input[Artifact],
    namespace: str,
    inference_url: Output[Artifact],
):
    import time
    from kubernetes import client as k8s_client, config as k8s_config
    MODEL_NAME = "csgo-match-predictor"

    with open(model_uri.path) as f:
        uri = f.read().strip()

    k8s_config.load_incluster_config()
    custom_api = k8s_client.CustomObjectsApi()

    body = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {"name": MODEL_NAME, "namespace": namespace},
        "spec": {
            "predictor": {
                "sklearn": {
                    "storageUri": uri,
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
            with open(inference_url.path, "w") as f:
                f.write(url)
            return
        time.sleep(5)
    raise TimeoutError(f"InferenceService {MODEL_NAME} not ready after 300s")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["requests==2.34.2", "pandas==2.2.3"],
)
def test_inference(
    inference_url: Input[Artifact],
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    n_samples: int = 10,
):
    import requests
    import pandas as pd
    MODEL_NAME = "csgo-match-predictor"

    with open(inference_url.path) as f:
        base_url = f.read().strip()

    X = pd.read_csv(X_test.path, index_col=0)
    y = pd.read_csv(y_test.path, index_col=0).squeeze()

    sample   = X.sample(n_samples, random_state=42)
    expected = y.loc[sample.index].tolist()

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
    packages_to_install=["pandas==2.2.3", "scikit-learn==1.8.0", "xgboost==3.2.0",
                         "joblib==1.5.3", "evidently==0.7.21"],
)
def monitoring(
    X_train: Input[Dataset],
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model: Input[Model],
    drift_threshold: float,
    drift_report: Output[Artifact],
    monitoring_metrics: Output[Metrics],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = pd.read_csv(X_train.path, index_col=0)
    current   = pd.read_csv(X_test.path,  index_col=0)
    y         = pd.read_csv(y_test.path,  index_col=0).squeeze()

    snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
    snapshot.save_html(drift_report.path)

    metrics_data  = snapshot.dict()["metrics"]
    drift_result  = metrics_data[0]["value"]
    share_drifted = drift_result["share"]
    n_drifted     = int(drift_result["count"])
    n_total       = len(reference.columns)

    print(f"Drifted columns: {n_drifted}/{n_total} ({share_drifted:.1%})")
    if share_drifted > drift_threshold:
        print(f"ALERT: drift {share_drifted:.1%} exceeds threshold {drift_threshold:.1%}")
    else:
        print("No significant drift detected.")

    pipeline = joblib.load(model.path)
    y_pred = pipeline.predict(current)
    y_prob = pipeline.predict_proba(current)[:, 1]
    acc    = accuracy_score(y, y_pred)
    auc    = roc_auc_score(y, y_prob)
    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    monitoring_metrics.log_metric("share_drifted",      share_drifted)
    monitoring_metrics.log_metric("n_drifted_columns",  float(n_drifted))
    monitoring_metrics.log_metric("accuracy",           acc)
    monitoring_metrics.log_metric("roc_auc",            auc)


# ── Pipeline DAG ──────────────────────────────────────────────────────────────

@dsl.pipeline(name="csgo-match-predictor")
def csgo_pipeline(
    model_type: str = "xgboost",
    train_ratio: float = 0.8,
    accuracy_threshold: float = 0.60,
    drift_threshold: float = 0.20,
    s3_endpoint: str = "http://seaweedfs.kubeflow.svc.cluster.local:8333",
    model_registry_url: str = "http://model-registry-service.csgo.svc.cluster.local:8080",
    deploy_namespace: str = "csgo",
    n_samples: int = 10,
):
    step00 = dvc_pull()

    step01 = feature_engineering(raw_data=step00.outputs["raw_data"])

    step02 = preprocessing(
        featured_data=step01.outputs["featured_data"],
        train_ratio=train_ratio,
    )

    step03 = train(
        X_train=step02.outputs["X_train"],
        y_train=step02.outputs["y_train"],
        model_type=model_type,
        namespace=deploy_namespace,
    ).after(step02)

    step04 = evaluate(
        model=step03.outputs["model"],
        X_test=step02.outputs["X_test"],
        y_test=step02.outputs["y_test"],
    )

    step05 = upload_model(
        model=step03.outputs["model"],
        eval_results=step04.outputs["eval_results"],
        accuracy_threshold=accuracy_threshold,
        s3_endpoint=s3_endpoint,
        model_registry_url=model_registry_url,
    )

    step06 = deploy(
        model_uri=step05.outputs["model_uri"],
        namespace=deploy_namespace,
    )

    step07 = test_inference(
        inference_url=step06.outputs["inference_url"],
        X_test=step02.outputs["X_test"],
        y_test=step02.outputs["y_test"],
        n_samples=n_samples,
    )

    step08 = monitoring(
        X_train=step02.outputs["X_train"],
        X_test=step02.outputs["X_test"],
        y_test=step02.outputs["y_test"],
        model=step03.outputs["model"],
        drift_threshold=drift_threshold,
    )

    for task in [step00, step01, step02, step03, step04, step05, step06, step07, step08]:
        add_node_selector(task, label_key="node-role.kubernetes.io/worker", label_value="worker")
        use_secret_as_env(
            task,
            secret_name="mlpipeline-minio-artifact",
            secret_key_to_env={"accesskey": "AWS_ACCESS_KEY_ID", "secretkey": "AWS_SECRET_ACCESS_KEY"},
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       action="store_true")
    parser.add_argument("--host",      default="http://localhost:3000")
    parser.add_argument("--namespace", default="csgo")
    args = parser.parse_args()

    compiler.Compiler().compile(csgo_pipeline, "pipeline.yaml")
    print("Compiled → pipeline.yaml")

    if args.run:
        import kfp
        client = kfp.Client(host=args.host, namespace=args.namespace)
        run = client.create_run_from_pipeline_func(
            csgo_pipeline,
            arguments={"model_type": "xgboost"},
            run_name="csgo-run",
            namespace=args.namespace,
        )
        print(f"Submitted → {run.run_id}")
