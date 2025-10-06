FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.15.0

VOLUME /mlruns

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlruns \
    --host 0.0.0.0 \
    --port 5000
