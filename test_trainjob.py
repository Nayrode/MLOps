"""
Test standalone du composant train : upload des données sur SeaweedFS,
crée un TrainJob v2, attend la fin, télécharge le modèle.

Usage:
    uv run python test_trainjob.py
    uv run python test_trainjob.py --namespace csgo --wait-timeout 300
"""
import argparse
import io
import time
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd


S3_ENDPOINT = "http://localhost:9000"   # via kubectl port-forward
S3_BUCKET   = "mlpipeline"
S3_KEY_ID   = "minio"
S3_SECRET   = "minio123"
NAMESPACE   = "csgo"


def s3_client(endpoint=S3_ENDPOINT):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=S3_KEY_ID,
        aws_secret_access_key=S3_SECRET,
    )


def make_test_data():
    """Génère des données de test identiques au format preprocessing."""
    maps = ["de_dust2", "de_inferno", "de_mirage"]
    rng  = np.random.default_rng(42)
    n    = 500

    X = pd.DataFrame({
        "elo_diff":         rng.normal(0, 100, n),
        "winrate_10_diff":  rng.uniform(-0.5, 0.5, n),
        "winrate_30_diff":  rng.uniform(-0.5, 0.5, n),
        "experience_diff":  rng.integers(-50, 50, n).astype(float),
        "rank_diff":        rng.integers(-10, 10, n).astype(float),
        "h2h_winrate":      rng.uniform(0, 1, n),
        "streak_diff":      rng.integers(-5, 5, n).astype(float),
        "map_winrate_diff": rng.uniform(-0.5, 0.5, n),
        "_map":             rng.choice(maps, n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="team_1_wins")
    return X, y


def upload_csv(s3, df, key):
    buf = io.BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  uploaded → s3://{S3_BUCKET}/{key}")


def create_trainjob(namespace, x_key, y_key, model_key,
                    n_estimators=10, learning_rate=0.05, max_depth=3):
    """Crée le TrainJob v2 xgboost-distributed (même code que pipeline.py)."""
    from kubernetes import client as k8s, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    s3_endpoint_cluster = "http://seaweedfs.kubeflow.svc.cluster.local:8333"

    training_script = """\
import subprocess, sys, os, io, tempfile
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "pandas==2.2.3", "scikit-learn==1.8.0", "xgboost==3.2.0",
    "joblib==1.5.3", "boto3==1.43.14"])
import boto3, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

s3 = boto3.client("s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

def load_csv(key):
    obj = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)

X = load_csv(os.environ["X_TRAIN_KEY"])
y = load_csv(os.environ["Y_TRAIN_KEY"]).squeeze()
print(f"Loaded X_train {X.shape}, y_train {y.shape}")

NUMERIC_COLS = ["elo_diff","winrate_10_diff","winrate_30_diff",
    "experience_diff","rank_diff","h2h_winrate","streak_diff","map_winrate_diff"]

pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("scaler",  StandardScaler(), NUMERIC_COLS),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["_map"]),
    ])),
    ("classifier", XGBClassifier(
        n_estimators=%(n_estimators)d, learning_rate=%(learning_rate)f,
        max_depth=%(max_depth)d, eval_metric="logloss", random_state=42, device="cuda",
    )),
])
pipeline.fit(X, y)
print(f"Train accuracy: {pipeline.score(X, y):.4f}")

with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    joblib.dump(pipeline, f.name)
    s3.upload_file(f.name, os.environ["S3_BUCKET"], os.environ["MODEL_KEY"])
print(f"Model saved → s3://{os.environ['S3_BUCKET']}/{os.environ['MODEL_KEY']}")
""" % {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}

    job_name = f"csgo-train-test-{int(time.time())}"
    custom_api = k8s.CustomObjectsApi()

    gpu_patch = {
        "manager": "test-trainjob",
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
                                        "containers": [{
                                            "name": "node",
                                            "resources": {
                                                "requests": {"nvidia.com/gpu": "1"},
                                                "limits":   {"nvidia.com/gpu": "1"},
                                            },
                                        }],
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
                "env": [
                    {"name": "S3_ENDPOINT",           "value": s3_endpoint_cluster},
                    {"name": "S3_BUCKET",             "value": S3_BUCKET},
                    {"name": "AWS_ACCESS_KEY_ID",     "value": S3_KEY_ID},
                    {"name": "AWS_SECRET_ACCESS_KEY", "value": S3_SECRET},
                    {"name": "X_TRAIN_KEY",           "value": x_key},
                    {"name": "Y_TRAIN_KEY",           "value": y_key},
                    {"name": "MODEL_KEY",             "value": model_key},
                ],
            },
        },
    }

    custom_api.create_namespaced_custom_object(
        group="trainer.kubeflow.org", version="v1alpha1",
        namespace=namespace, plural="trainjobs", body=body,
    )
    print(f"TrainJob '{job_name}' soumis dans '{namespace}'.")
    print(f"  kubectl logs -n {namespace} -l trainer.kubeflow.org/trainjob={job_name} -f")
    return job_name, custom_api


def wait_for_trainjob(custom_api, namespace, job_name, timeout=300):
    print(f"Attente du TrainJob '{job_name}'...")
    interval, elapsed = 10, 0
    while elapsed < timeout:
        job = custom_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", name=job_name,
        )
        for js in job.get("status", {}).get("jobsStatus", []):
            if js.get("name") == "node":
                active    = js.get("active", 0)
                succeeded = js.get("succeeded", 0)
                failed    = js.get("failed", 0)
                print(f"  [{elapsed:3d}s] active={active} succeeded={succeeded} failed={failed}")
                if succeeded >= 1:
                    print("TrainJob terminé avec succès.")
                    return True
                if failed >= 1:
                    print("TrainJob a échoué.")
                    return False
        time.sleep(interval)
        elapsed += interval
    print(f"Timeout après {timeout}s.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",     default=NAMESPACE)
    parser.add_argument("--wait-timeout",  type=int, default=700)
    parser.add_argument("--s3-port",       type=int, default=9000,
                        help="port-forward local vers SeaweedFS")
    args = parser.parse_args()

    local_endpoint = f"http://localhost:{args.s3_port}"
    s3 = s3_client(local_endpoint)

    # Crée le bucket si nécessaire
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if S3_BUCKET not in existing:
        s3.create_bucket(Bucket=S3_BUCKET)
        print(f"Bucket '{S3_BUCKET}' créé.")

    print("1. Upload des données de test sur SeaweedFS...")
    X, y = make_test_data()
    ts      = int(time.time())
    x_key   = f"csgo/test-data/{ts}/X_train.csv"
    y_key   = f"csgo/test-data/{ts}/y_train.csv"
    model_key = f"csgo/test-data/{ts}/model.pkl"
    upload_csv(s3, X, x_key)
    upload_csv(s3, y, y_key)

    print("\n2. Soumission du TrainJob...")
    job_name, custom_api = create_trainjob(
        namespace=args.namespace,
        x_key=x_key, y_key=y_key, model_key=model_key,
        n_estimators=10,   # rapide pour le test
    )

    print("\n3. Attente...")
    ok = wait_for_trainjob(custom_api, args.namespace, job_name, args.wait_timeout)

    if ok:
        print("\n4. Téléchargement du modèle...")
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            s3.download_file(S3_BUCKET, model_key, f.name)
            import joblib
            model = joblib.load(f.name)
            print(f"  Modèle chargé : {type(model).__name__}")
            print("Test OK.")
    else:
        print(f"\nPour voir les logs du job :")
        print(f"  kubectl logs -n {args.namespace} -l trainer.kubeflow.org/trainjob={job_name} -f")
