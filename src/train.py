# src/train.py
# Pipeline construction, model training, and model persistence.

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS_PATH
from src.utils import ensure_dir


def build_pipelines(tfidf_params: dict) -> dict[str, Pipeline]:
    """Return a dict of five unfitted sklearn Pipelines keyed by model name."""
    return {
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB()),
        ]),
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "svm": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(max_iter=1000)),
        ]),
        "decision_tree": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", DecisionTreeClassifier()),
        ]),
        "random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", RandomForestClassifier(n_estimators=100)),
        ]),
    }


def train_model(pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Fit pipeline on training data and return the fitted pipeline."""
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, model_name: str) -> None:
    """Serialize fitted pipeline to models/{model_name}.joblib."""
    ensure_dir(MODELS_PATH)
    path = f"{MODELS_PATH}{model_name}.joblib"
    joblib.dump(pipeline, path)
    print(f"  Saved: {path}")
