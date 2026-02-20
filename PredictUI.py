# API version
import streamlit as st
import requests

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8002/predict"

st.set_page_config(page_title="AI Risk Monitor", layout="centered")

st.title("🚨 AI Risk Monitoring System")
st.write("Streamlit Frontend → FastAPI Backend (Port 8002)")

user_text = st.text_area("Enter scenario text:")

if st.button("Analyze Risk"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Calling API..."):
                response = requests.post(
                    API_URL,
                    json={"text": user_text}
                )

            if response.status_code == 200:
                result = response.json()

                st.subheader("🔎 TF-IDF Model")
                st.metric("Risk Score", result["tfidf"]["risk_score"])
                st.write("Level:", result["tfidf"]["risk_level"])
               

                st.subheader("🧠 LLM Model")
                st.metric("Risk Score", result["llm"]["risk_score"])
                st.write("Level:", result["llm"]["risk_level"])
                

                st.subheader("📊 Model Difference")
                st.write("Score Difference:", result["difference"])

                if result["difference"] > 20:
                    st.warning("⚠ Significant disagreement between models.")

            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Connection Error: {e}")
