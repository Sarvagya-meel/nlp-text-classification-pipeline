# src/evaluate.py
# Metric computation, over/underfitting diagnosis, and plot/report persistence.

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import learning_curve

from src.config import METRICS_PATH, FIGURES_PATH
from src.utils import ensure_dir


def evaluate_model(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    """Return accuracy, precision, recall, and F1 for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }


def compute_train_test_scores(
    pipeline: Pipeline,
    X_train: pd.Series, y_train: pd.Series,
    X_test: pd.Series,  y_test: pd.Series,
) -> dict:
    """Return train and test accuracy scores for over/underfitting diagnosis."""
    return {
        "train_score": round(pipeline.score(X_train, y_train), 4),
        "test_score":  round(pipeline.score(X_test, y_test), 4),
    }


def diagnose_fit(train_score: float, test_score: float, threshold: float = 0.10) -> str:
    """Return 'overfit', 'underfit', or 'good_fit' based on train/test gap."""
    if train_score < 0.70 and test_score < 0.70:
        return "underfit"
    if train_score - test_score > threshold:
        return "overfit"
    return "good_fit"


def save_metrics(metrics: dict, model_name: str) -> None:
    """Write metrics dict as JSON to reports/metrics/{model_name}_metrics.json."""
    ensure_dir(METRICS_PATH)
    path = f"{METRICS_PATH}{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {path}")


def plot_confusion_matrix(
    pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, model_name: str
) -> None:
    """Save a seaborn confusion matrix heatmap to reports/figures/."""
    ensure_dir(FIGURES_PATH)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=pipeline.classes_ if hasattr(pipeline, "classes_") else "auto",
                yticklabels=pipeline.classes_ if hasattr(pipeline, "classes_") else "auto")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    path = f"{FIGURES_PATH}{model_name}_confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_learning_curve(
    pipeline: Pipeline, X: pd.Series, y: pd.Series, model_name: str
) -> None:
    """Save a learning curve plot (train vs CV score) to reports/figures/."""
    ensure_dir(FIGURES_PATH)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=3, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1,
    )
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train score")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="CV score")
    plt.title(f"Learning Curve — {model_name}")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}{model_name}_learning_curve.png"
    plt.savefig(path)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_metrics_comparison(all_metrics: dict[str, dict]) -> None:
    """Save a grouped bar chart comparing all models across all four metrics."""
    ensure_dir(FIGURES_PATH)
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    models = list(all_metrics.keys())
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics_keys):
        values = [all_metrics[m].get(metric, 0) for m in models]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}metrics_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Plot saved: {path}")
