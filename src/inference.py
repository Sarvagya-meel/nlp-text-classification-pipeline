# src/inference.py
# Load a saved model and run predictions on new text input.

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.config import MODELS_PATH
from src.preprocessing import clean_text


def load_model(model_name: str) -> Pipeline:
    """Load and return a fitted pipeline from models/{model_name}.joblib."""
    path = f"{MODELS_PATH}{model_name}.joblib"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No saved model found at: {path}")


def predict(text: str, model_name: str) -> tuple[str, float]:
    """Preprocess text and return (predicted_label, confidence_score).

    Confidence is derived from predict_proba where available,
    or from the normalized decision function for LinearSVC.
    """
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    pipeline = load_model(model_name)
    cleaned = clean_text(text)

    # Confidence: use predict_proba if available, else decision function
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([cleaned])[0]
        label = pipeline.classes_[np.argmax(proba)]
        confidence = float(np.max(proba))
    else:
        # LinearSVC: normalize decision scores to [0, 1]
        decision = pipeline.decision_function([cleaned])[0]
        if decision.ndim == 0:
            decision = np.array([decision])
        scores = np.exp(decision) / np.sum(np.exp(decision))  # softmax
        label = pipeline.classes_[np.argmax(scores)]
        confidence = float(np.max(scores))

    return str(label), round(confidence, 4)
