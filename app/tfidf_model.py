import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# -----------------------------
# SAFE PROJECT PATH (IMPORTANT)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "train.parquet"

# Load data
df = pd.read_parquet(data_path)

# Input text (combine columns for better learning)
df["text_input"] = df["context"] + " " + df["query"]

X = df["text_input"]
y = df["risk_score"]

# -----------------------------
# 2. Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Build pipeline
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("regressor", LinearRegression())
])

# -----------------------------
# 4. Train model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error:", mae)

# -----------------------------
# 6. Save model
# -----------------------------
model_path = BASE_DIR / "app" / "risk_model.pkl"
joblib.dump(model, model_path)
print("Model saved successfully!")
