from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent  # app/
tfidf_model_path = BASE_DIR / "risk_model.pkl"
llm_model_path = BASE_DIR / "risk_llm_model.pkl"
embedder_path = BASE_DIR / "embedder.pkl"
log_path = BASE_DIR / "predictions_log.csv"

# -----------------------------
# Load models
# -----------------------------
tfidf_model = joblib.load(tfidf_model_path)
llm_model = joblib.load(llm_model_path)
embedder = joblib.load(embedder_path)

# -----------------------------
# Create API
# -----------------------------
app = FastAPI(title="AI Risk Monitoring Compare API")

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
    return {"message": "AI Risk Monitoring Compare API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_risk(data: RiskInput):
    text = data.text

    # -----------------------------
    # TF-IDF prediction
    # -----------------------------
    tfidf_score = round(float(tfidf_model.predict([text])[0]), 2)

    if tfidf_score < 40:
        tfidf_level = "LOW"
    elif tfidf_score < 70:
        tfidf_level = "MEDIUM"
    else:
        tfidf_level = "HIGH"

    # -----------------------------
    # LLM prediction
    # -----------------------------
    embedding = embedder.encode([text])
    llm_score = round(float(llm_model.predict(embedding)[0]), 2)

    if llm_score < 40:
        llm_level = "LOW"
    elif llm_score < 70:
        llm_level = "MEDIUM"
    else:
        llm_level = "HIGH"

    # -----------------------------
    # Difference
    # -----------------------------
    score_diff = round(abs(tfidf_score - llm_score), 2)

    # -----------------------------
    # Escalation Logic
    # -----------------------------
    escalation_flag = "YES" if (
        tfidf_level == "HIGH" or llm_level == "HIGH"
    ) else "NO"

    agreement_flag = "DISAGREE" if tfidf_level != llm_level else "AGREE"

    # -----------------------------
    # Logging for Power BI
    # -----------------------------
    new_entry = pd.DataFrame({
        "timestamp": [datetime.now()],
        "text": [text],
        "tfidf_score": [tfidf_score],
        "tfidf_level": [tfidf_level],
        "llm_score": [llm_score],
        "llm_level": [llm_level],
        "difference": [score_diff],
        "escalation": [escalation_flag],
        "model_agreement": [agreement_flag]
    })

    if log_path.exists():
        new_entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(log_path, index=False)

    # -----------------------------
    # API Response
    # -----------------------------
    return {
        "tfidf": {
            "risk_score": tfidf_score,
            "risk_level": tfidf_level
        },
        "llm": {
            "risk_score": llm_score,
            "risk_level": llm_level
        },
        "difference": score_diff,
        "escalation": escalation_flag,
        "agreement": agreement_flag
    }
