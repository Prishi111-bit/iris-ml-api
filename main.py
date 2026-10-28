
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "models.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
classes = artifact["classes"]

app = FastAPI(title="Iris Flower Prediction API")


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.get("/")
def home():
    return {"message": "Iris API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(flower: FlowerInput):
    values = {
        "sepal_length": flower.sepal_length,
        "sepal_width": flower.sepal_width,
        "petal_length": flower.petal_length,
        "petal_width": flower.petal_width,
    }

    input_data = pd.DataFrame([values], columns=features)
    prediction = str(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    return {
        "prediction": prediction,
        "confidence": round(float(max(probabilities)), 4),
    }
