"""Kubeflow Pipelines v2 — CS:GO match predictor pipeline.

Build & push the image first:
    docker build -t YOUR_REGISTRY/csgo-mlops:latest .
    docker push YOUR_REGISTRY/csgo-mlops:latest

Compile:
    pip install kfp
    python pipeline.py          # writes pipeline.yaml

Submit (once Kubeflow is running):
    kubectl apply -f pvc.yaml
    python pipeline.py --run    # compiles + submits to KFP
"""
import argparse
from kfp import dsl, compiler

IMAGE     = "YOUR_REGISTRY/csgo-mlops:latest"
DATA_DIR  = "/data"
MODEL_DIR = "/data"


# ── Shared volume (PVC) ──────────────────────────────────────────────────────

pvc = dsl.PipelineVolume(pvc_name="csgo-data-pvc")


# ── Components ───────────────────────────────────────────────────────────────

def make_step(name: str, cmd: str) -> dsl.ContainerOp:
    """Helper: one container step writing to the shared PVC."""
    op = dsl.ContainerOp(
        name=name,
        image=IMAGE,
        command=["bash", "-c"],
        arguments=[cmd],
        pvolumes={DATA_DIR: pvc},
    )
    op.container.set_memory_request("512Mi")
    return op


# ── Pipeline DAG ─────────────────────────────────────────────────────────────

@dsl.pipeline(
    name="csgo-match-predictor",
    description="Feature engineering → preprocessing → train → evaluate → monitoring",
)
def csgo_pipeline(
    model_type: str = "xgboost",
    mlflow_uri: str = "",
):
    # Step 01 — Feature engineering
    fe = make_step(
        "feature-engineering",
        f"cd /app && python 01_feature_engineering/feature_engineering.py",
    )

    # Step 02 — Preprocessing
    pre = make_step(
        "preprocessing",
        f"cd /app && python 02_preprocessing/preprocessing.py",
    ).after(fe)

    # Step 03 — Train
    train = make_step(
        "train",
        f"cd /app && DATA_DIR={DATA_DIR} MODEL_DIR={MODEL_DIR} "
        f"MODEL_TYPE={model_type} MLFLOW_TRACKING_URI={mlflow_uri} "
        f"python 03_train_model_kubernetes/train.py",
    ).after(pre)

    # Step 04 — Evaluate
    evaluate = make_step(
        "evaluate",
        f"cd /app && MLFLOW_TRACKING_URI={mlflow_uri} "
        f"python 04_evaluate_model/evaluate_model.py",
    ).after(train)

    # Step 09 — Monitoring
    make_step(
        "monitoring",
        f"cd /app && python 09_model_monitoring/model_monitoring.py",
    ).after(evaluate)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Submit to KFP after compiling")
    parser.add_argument("--host", default="http://localhost:3000", help="KFP host URL")
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
