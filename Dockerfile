FROM python:3.11-slim
WORKDIR /app

# All pipeline steps + inference
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    pandas scikit-learn xgboost joblib \
    fastapi uvicorn evidently \
    mlflow python-dotenv

# Copy all step scripts and shared modules
COPY 01_feature_engineering/feature_engineering.py 01_feature_engineering/
COPY 02_preprocessing/preprocessing.py             02_preprocessing/
COPY 03_train_model_kubernetes/train.py            03_train_model_kubernetes/
COPY 04_evaluate_model/evaluate_model.py           04_evaluate_model/
COPY 09_model_monitoring/model_monitoring.py       09_model_monitoring/
COPY predict.py serve.py ./
