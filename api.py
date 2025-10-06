from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn

app = FastAPI()

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "mlops_lab"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' does not exist in MLflow.")

# Get the latest run ID
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"])
if runs.empty:
    raise ValueError(f"No runs found for experiment '{EXPERIMENT_NAME}'.")

latest_run_id = runs.iloc[0].run_id
MODEL_URI = f"runs:/{latest_run_id}/model"

# Load the model
model = mlflow.sklearn.load_model(MODEL_URI)

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}
