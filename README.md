# AI Risk Severity Scoring & Model Comparison System

## Project Overview
This project demonstrates an end-to-end AI-powered risk scoring and monitoring system, combining:
- Traditional NLP (TF-IDF + ML model)
- Transformer-based embeddings (LLM)
- FastAPI backend for model serving
- Streamlit frontend for interactive testing
- Power BI dashboard for monitoring and governance analytics

The system simulates a high-risk operational environment where textual inputs are scored for severity and compared across two different NLP modeling approaches.

---

## Problem Statement
In high-risk environments (e.g., financial alerts, operational incidents, compliance signals), organizations must:
- Assign severity scores to textual cases
- Compare model predictions
- Detect model disagreement
- Escalate critical cases
- Monitor risk trends over time

This project simulates such a scenario by building a hybrid NLP risk scoring pipeline and exposing it through an API with governance tracking.

---

## System Architecture
```text
User Input (Streamlit UI)
        ↓
FastAPI Backend
        ↓
TF-IDF Model      LLM Embedding Model
        ↓              ↓
Risk Score Comparison
        ↓
Escalation & Agreement Logic
        ↓
Prediction Logging (CSV)
        ↓
Power BI Dashboard (Monitoring Layer)
