from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

# Load latest model from MLflow
model = mlflow.pyfunc.load_model("runs:/<REPLACE_WITH_RUN_ID>/model")

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    input_data = [[data.feature1, data.feature2]]
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
