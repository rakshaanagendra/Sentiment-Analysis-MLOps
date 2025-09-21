# app.py
import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Path to local model (relative to repo root)
MODEL_PATH = os.path.join("outputs", "model", "logreg_tfidf_pipeline.pkl")

def load_pipeline(path: str):
    """
    Load scikit-learn pipeline from local pickle file.
    Runs once at startup.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

# Create FastAPI app
app = FastAPI(title="IMDB Sentiment Analysis API", version="1.0")

# Request schema
class PredictRequest(BaseModel):
    text: str

# Response schema
class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float

# Load model at startup
try:
    pipeline = load_pipeline(MODEL_PATH)
except Exception as e:
    pipeline = None
    load_error = str(e)

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict sentiment for input text"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    texts = [req.text]
    try:
        pred = int(pipeline.predict(texts)[0])
        confidence = float(pipeline.predict_proba(texts).max())

        label_map = {0: "negative", 1: "positive"}
        label = label_map.get(pred, "unknown")

        return PredictResponse(prediction=pred, label=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Welcome to IMDB Sentiment Analysis API. Go to /docs for Swagger UI."}

