FROM python:3.11-slim
WORKDIR /app
COPY train.py .
RUN pip install --no-cache-dir scikit-learn pandas mlflow joblib
CMD ["python", "train.py"]
