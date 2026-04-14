from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import EmployeeFeatures, PredictionResponse, TrainingResponse
from src.config import DATASET_PATH, METRICS_PATH, MODEL_PATH
from src.data_generator import save_dataset
from src.modeling import PerformancePredictor
from src.predict import predict_single
from src.visualize import create_class_distribution_plot, create_feature_importance_plot

app = FastAPI(title="Employee Performance Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "Employee Performance Predictor API",
        "train_endpoint": "/train",
        "predict_endpoint": "/predict",
        "docs": "/docs",
    }


@app.post("/train", response_model=TrainingResponse)
def train_model() -> TrainingResponse:
    df = save_dataset(num_rows=1200, random_state=42)
    predictor = PerformancePredictor()
    metrics, _, _, _ = predictor.train(df)
    create_class_distribution_plot(df)
    create_feature_importance_plot(predictor)
    return TrainingResponse(
        message="Model trained successfully",
        dataset_rows=len(df),
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
    )


@app.get("/health")
def health() -> dict:
    return {
        "dataset_exists": DATASET_PATH.exists(),
        "model_exists": MODEL_PATH.exists(),
        "metrics_exists": METRICS_PATH.exists(),
    }


@app.get("/metrics")
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/sample-data")
def sample_data(limit: int = 10) -> list[dict]:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Train the model first.")
    df = pd.read_csv(DATASET_PATH)
    return df.head(limit).to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: EmployeeFeatures) -> PredictionResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not found. Please call /train first.")
    result = predict_single(features.model_dump())
    return PredictionResponse(**result)


@app.get("/artifacts")
def artifacts() -> dict:
    base = Path(__file__).resolve().parent.parent
    return {
        "dataset": str((base / "data" / "employee_data.csv").resolve()),
        "model": str((base / "models" / "employee_performance_model.pkl").resolve()),
        "metrics": str((base / "outputs" / "metrics.json").resolve()),
        "plots": [
            str((base / "outputs" / "class_distribution.png").resolve()),
            str((base / "outputs" / "feature_importance.png").resolve()),
        ],
    }
