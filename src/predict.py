import os
import joblib
import numpy as np
from src.train import train

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

model = None

def load_model():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Training model...")
            train()

        model = joblib.load(MODEL_PATH)

    return model


def predict_workforce(features):
    model = load_model()
    arr = np.array(features).reshape(1, -1)
    result = model.predict(arr)
    return float(result[0])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")

joblib.dump(model, model_path)