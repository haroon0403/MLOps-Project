from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os

app = FastAPI()

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_URI = "runs:/<REPLACE_WITH_RUN_ID>/model"  # Replace manually after training

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}
