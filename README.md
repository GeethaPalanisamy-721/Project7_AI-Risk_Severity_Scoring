**AI Risk Severity Scoring & Model Comparison System

Project Overview

This project demonstrates an end-to-end AI-powered risk scoring and monitoring system, combining:
	• Traditional NLP (TF-IDF + ML model)
	• Transformer-based embeddings (LLM)
	• FastAPI backend for model serving
	• Streamlit frontend for interactive testing
	• Power BI dashboard for monitoring and governance analytics
The system simulates a high-risk operational environment where textual inputs are scored for severity and compared across two different NLP modeling approaches.

Problem Statement

In high-risk environments (e.g., financial alerts, operational incidents, compliance signals), organizations must:
	• Assign severity scores to textual cases
	• Compare model predictions
	• Detect model disagreement
	• Escalate critical cases
	• Monitor risk trends over time
This project simulates such a scenario by building a hybrid NLP risk scoring pipeline and exposing it through an API with governance tracking.

System Architecture

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

Machine Learning Pipeline

Model 1: TF-IDF + Traditional ML
	• Text vectorized using TF-IDF
	• Trained classification/regression model
	• Outputs risk severity score (0–100)
Purpose:
	• Baseline interpretable NLP model

Model 2: LLM Embedding + ML Model
	• Sentence embeddings generated via transformer-based embedder
	• Separate ML model trained on embeddings
	• Outputs independent risk score
Purpose:
	• Compare classical NLP vs modern embedding-based representation

Model Comparison Logic

For every input:
	• Compute TF-IDF score
	• Compute LLM-based score
	• Calculate score difference
	• Detect model agreement/disagreement
	• Flag escalation if severity is HIGH
This introduces model governance monitoring, not just prediction.

Backend: FastAPI
The backend is built using FastAPI.
Features:
	• /predict endpoint
	• JSON-based inference
	• Structured response schema
	• Logging layer for monitoring
	• Production-ready model loading using joblib
Why FastAPI?
	• High performance
	• Async-ready
	• Industry-standard for ML serving

Frontend: Streamlit
Streamlit is used to:
	• Provide interactive text input
	• Display dual model scores
	• Show severity classification
	• Display escalation & agreement flags
This layer simulates a lightweight internal risk review tool.

Monitoring & Analytics: Power BI

The system logs each prediction into: predictions_log.csv
This file serves as the monitoring dataset.
Power BI dashboard includes:
	• Total Cases
	• High Risk %
	• Escalation %
	• Model Agreement %
	• Average Score Difference
	• Severity distribution
	• Score trend over time
This simulates enterprise model governance monitoring.
The dashboard auto-updates when new rows are appended and refreshed.

Logged Data Schema

Each prediction stores:
	• timestamp
	• input text
	• tfidf_score
	• tfidf_level
	• llm_score
	• llm_level
	• score difference
	• escalation flag
	• model agreement flag

Key Concepts Demonstrated

	• NLP Feature Engineering
	• Transformer Embeddings
	• Multi-model comparison
	• ML API serving
	• Frontend-backend integration
	• Monitoring pipeline design
	• Governance-aware architecture
	• BI-based model analytics

Tech Stack

	• Python
	• scikit-learn
	• Transformer Embeddings
	• FastAPI
	• Streamlit
	• Power BI
	• Pandas
	• Joblib

Why This Project Matters1:

This is not just a model.
It demonstrates understanding of:
	• ML pipeline lifecycle
	• Serving models in production
	• Logging and traceability
	• Model disagreement analysis
	• Governance and escalation logic
	• Business-facing dashboards
The project bridges:
Data Science → Backend Engineering → BI Analytics.

How to Run:

1.Install Dependencies
pip install -r requirements.txt
2.Run API
uvicorn app.main_compare:app --reload --port 8002
3.Run Streamlit
streamlit run streamlit_app.py
4.Open Power BI
Load predictions_log.csv and refresh.

Author Perspective

This project was built to simulate a realistic ML system that goes beyond model training and demonstrates:
	• End-to-end system thinking
	• Production considerations
	• Monitoring mindset
	• Business-aligned analytics
It reflects how AI systems operate in controlled, high-risk environments where severity scoring and governance matter.


