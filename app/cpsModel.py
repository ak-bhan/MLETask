import json
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from app.utils import ensure_model_directory

CPS_MODEL_DIR = "model_store_cps"
CPS_MODEL_PATH = os.path.join(CPS_MODEL_DIR, "cps_model.pkl")

_loaded_cps_model = None

def train_cps_model_from_big_file(big_json_file_path: str):
    headlines = []
    categories = []
    with open(big_json_file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            headline = record.get("headline", "")
            category = record.get("category", "")

            if headline and category:
                headlines.append(headline)
                categories.append(category)

    df = pd.DataFrame({"headline": headlines, "category": categories})
    print(f"Loaded {len(df)} samples.")

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["headline"])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])

    model = LogisticRegression(max_iter=300)
    model.fit(X, y)

    ensure_model_directory(CPS_MODEL_DIR)
    with open(CPS_MODEL_PATH, "wb") as f:
        pickle.dump((vectorizer, label_encoder, model), f)

    global _loaded_cps_model
    _loaded_cps_model = (vectorizer, label_encoder, model)

    print("CPS Model trained and saved successfully.")

def load_cps_model_if_exists():
    global _loaded_cps_model
    if os.path.exists(CPS_MODEL_PATH):
        with open(CPS_MODEL_PATH, "rb") as f:
            _loaded_cps_model = pickle.load(f)

def predict_cps(text: str) -> str:
    if _loaded_cps_model is None:
        raise Exception("CPS Model not loaded.")
    vectorizer, label_encoder, model = _loaded_cps_model
    X = vectorizer.transform([text])
    y_pred_encoded = model.predict(X)[0]
    category = label_encoder.inverse_transform([y_pred_encoded])[0]
    return category
