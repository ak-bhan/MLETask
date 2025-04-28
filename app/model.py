import pickle
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from app.utils import ensure_model_directory

MODEL_DIR = "model_store"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

_loaded_model = None

def train(samples):
    df = pd.DataFrame(samples)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = LogisticRegression(max_iter=200)
    model.fit(X, y_encoded)

    ensure_model_directory(MODEL_DIR)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, label_encoder), f)

    global _loaded_model
    _loaded_model = (model, label_encoder)

def load_model_if_exists():
    global _loaded_model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _loaded_model = pickle.load(f)

def predict(features):
    if _loaded_model is None:
        raise Exception("Model is not trained yet.")
    model, label_encoder = _loaded_model
    X = pd.DataFrame([features])
    prediction_encoded = model.predict(X)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    return prediction_label
