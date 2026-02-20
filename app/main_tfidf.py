from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

# -----------------------------
# Load saved model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "app" / "risk_model.pkl"

model = joblib.load(model_path)

# -----------------------------
# Create API
# -----------------------------
app = FastAPI(title="AI Risk Monitoring API")

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
    return {"message": "AI Risk Monitoring API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_risk(data: RiskInput):

    prediction = model.predict([data.text])[0]
    score = round(float(prediction), 2)

    # -----------------------------
    # Risk Level Logic
    # -----------------------------
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
