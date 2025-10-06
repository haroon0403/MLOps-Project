from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import time

app = FastAPI()   # Must be at top-level

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "mlops_lab"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None

def load_model():
    global model
    for i in range(10):
        try:
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                raise ValueError(f"Experiment '{EXPERIMENT_NAME}' does not exist")
            runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time desc"])
            if runs.empty:
                raise ValueError(f"No runs found for experiment '{EXPERIMENT_NAME}'")
            latest_run_id = runs.iloc[0].run_id
            model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
            print("Model loaded successfully")
            return
        except Exception as e:
            print(f"MLflow not ready yet: {e}, retrying in 5s...")
            time.sleep(5)
    raise RuntimeError("Failed to load model from MLflow")

@app.on_event("startup")
def startup_event():
    load_model()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}
