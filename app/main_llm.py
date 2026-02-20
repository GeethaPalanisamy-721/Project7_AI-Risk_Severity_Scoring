from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "risk_llm_model.pkl"
embedder_path = BASE_DIR / "embedder.pkl"

# -----------------------------
# Load model and embedder
# -----------------------------
model = joblib.load(model_path)
embedder = joblib.load(embedder_path)

# -----------------------------
# Create API
# -----------------------------
app = FastAPI(title="AI Risk Monitoring LLM API")

# -----------------------------
# Input schema
# -----------------------------
class RiskInput(BaseModel):
    text: str

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "AI Risk Monitoring LLM API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_risk(data: RiskInput):
    # Convert text to embedding
    embedding = embedder.encode([data.text])
    
    # Predict risk score
    prediction = model.predict(embedding)[0]
    score = round(float(prediction), 2)

    # Risk Level Logic
    if score < 40:
        level = "LOW"
        message = "Low risk scenario"
    elif score < 70:
        level = "MEDIUM"
        message = "Moderate risk — review recommended"
    else:
        level = "HIGH"
        message = "High risk scenario detected"

    return {
        "risk_score": score,
        "risk_level": level,
        "message": message
    }
