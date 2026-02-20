import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "train.parquet"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(data_path)

# Combine text columns
df["text_input"] = df["context"] + " " + df["query"]

X = df["text_input"]
y = df["risk_score"]

# -----------------------------
# Load embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

# Text → semantic vectors
X_embeddings = embedder.encode(
    X.tolist(),
    show_progress_bar=True
)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("LLM-ready MAE:", mae)

# -----------------------------
# Save model + embedder
# -----------------------------
joblib.dump(model, BASE_DIR / "app" / "risk_llm_model.pkl")
joblib.dump(embedder, BASE_DIR / "app" / "embedder.pkl")

print("LLM-ready model saved!")
