import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Read dataset
df = pd.read_csv("data/data.csv")
X = df.drop("target", axis=1)
y = df["target"]

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("mlops_lab")

with mlflow.start_run():
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

print("Training done and model logged to MLflow.")
