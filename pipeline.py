"""Kubeflow Pipelines v2 — CS:GO match predictor.

Each step runs in its own Pod, sharing data through a PVC at /app/data.
results.csv is baked into the image and copied to the PVC by the first step.

Usage:
    uv run python pipeline.py              # compile → pipeline.yaml
    uv run python pipeline.py --run        # compile + submit to KFP
    uv run python pipeline.py --run --host http://<kfp-host>
"""
import argparse

from kfp import compiler, dsl
from kfp.dsl import ConcatPlaceholder
from kfp.kubernetes import CreatePVC, DeletePVC, mount_pvc

IMAGE     = "popopolette/csgo-mlops:latest"
DATA_PATH = "/app/data"        # PVC mount — ../data relative paths resolve here
STATIC    = "/app/data_static" # results.csv baked into the image


# ── Components ───────────────────────────────────────────────────────────────

@dsl.container_component
def feature_engineering():
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[f"cp {STATIC}/results.csv {DATA_PATH}/results.csv && "
               "cd /app/01_feature_engineering && python feature_engineering.py"],
    )


@dsl.container_component
def preprocessing():
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=["cd /app/02_preprocessing && python preprocessing.py"],
    )


@dsl.container_component
def train(model_type: str, mlflow_uri: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[ConcatPlaceholder([
            f"export DATA_DIR={DATA_PATH} MODEL_DIR={DATA_PATH} MODEL_TYPE=",
            model_type,
            " MLFLOW_TRACKING_URI=",
            mlflow_uri,
            " && cd /app/03_train_model_kubernetes && python train.py",
        ])],
    )


@dsl.container_component
def evaluate(mlflow_uri: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[ConcatPlaceholder([
            "export MLFLOW_TRACKING_URI=",
            mlflow_uri,
            f" DATA_DIR={DATA_PATH} && cd /app/04_evaluate_model && python evaluate_model.py",
        ])],
    )


@dsl.container_component
def monitoring():
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[f"DATA_DIR={DATA_PATH} cd /app/09_model_monitoring && python model_monitoring.py"],
    )


# ── Pipeline DAG ─────────────────────────────────────────────────────────────

@dsl.pipeline(name="csgo-match-predictor")
def csgo_pipeline(model_type: str = "xgboost", mlflow_uri: str = ""):

    pvc = CreatePVC(
        pvc_name="csgo-data-pvc",
        access_modes=["ReadWriteMany"],
        size="2Gi",
        storage_class_name="nfs-rwx",
    )

    def with_data(task):
        mount_pvc(task, pvc_name=pvc.outputs["name"], mount_path=DATA_PATH)
        return task

    fe  = with_data(feature_engineering())
    pre = with_data(preprocessing().after(fe))
    tr  = with_data(train(model_type=model_type, mlflow_uri=mlflow_uri).after(pre))
    ev  = with_data(evaluate(mlflow_uri=mlflow_uri).after(tr))
    mon = with_data(monitoring().after(ev))

    DeletePVC(pvc_name=pvc.outputs["name"]).after(mon)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  action="store_true", help="Submit to KFP after compiling")
    parser.add_argument("--host", default="http://localhost:3000", help="KFP UI host")
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
        print(f"Submitted → run id: {run.run_id}")
