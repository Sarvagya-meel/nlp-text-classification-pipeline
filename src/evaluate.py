# src/evaluate.py
# Metric computation, over/underfitting diagnosis, and plot/report persistence.
#
# ─────────────────────────────────────────────────────────────────────────────
# METRIC REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY  — Out of all predictions, how many were correct?
#             Formula : (TP + TN) / (TP + TN + FP + FN)
#             Good for: balanced datasets. Misleading on imbalanced ones.
#
# PRECISION — Out of all POSITIVE predictions, how many were actually positive?
#             Formula : TP / (TP + FP)
#             Good for: when false positives are costly (e.g. spam filter).
#
# RECALL    — Out of all ACTUAL positives, how many did we correctly catch?
#             Formula : TP / (TP + FN)
#             Good for: when false negatives are costly (e.g. disease detection).
#
# F1 SCORE  — Harmonic mean of Precision and Recall. Balances both.
#             Formula : 2 * (Precision * Recall) / (Precision + Recall)
#             Good for: imbalanced datasets where both FP and FN matter.
#
# All three (Precision, Recall, F1) are macro-averaged — each class is
# weighted equally regardless of how many samples it has.
# ─────────────────────────────────────────────────────────────────────────────
#
# OVERFITTING vs UNDERFITTING
# ─────────────────────────────────────────────────────────────────────────────
#
#  OVERFITTING
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ Train score: HIGH (e.g. 0.99)   Test score: LOW (e.g. 0.65)            │
#  │ Gap > 0.10 → model memorised training data, fails on unseen examples   │
#  │                                                                         │
#  │ WHY it happens:                                                         │
#  │  - Model is too complex (e.g. deep Decision Tree with no max_depth)    │
#  │  - Too many features relative to training samples                      │
#  │  - Training data is too small or not representative                    │
#  │                                                                         │
#  │ HOW to fix:                                                             │
#  │  - Add regularisation (LogReg: lower C; SVM: lower C)                  │
#  │  - Reduce max_features in TF-IDF to shrink feature space               │
#  │  - Set max_depth on DecisionTree / min_samples_leaf on RandomForest    │
#  │  - Collect more training data                                           │
#  │  - Use cross-validation instead of a single train/test split           │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  UNDERFITTING
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ Train score: LOW (e.g. 0.60)    Test score: LOW (e.g. 0.58)            │
#  │ Both below 0.70 → model too simple, not capturing the signal           │
#  │                                                                         │
#  │ WHY it happens:                                                         │
#  │  - Model is too simple for the task (e.g. NB on complex patterns)      │
#  │  - Insufficient features (too few n-grams, max_features too low)       │
#  │  - Over-aggressive preprocessing removed meaningful tokens             │
#  │  - Too little training data                                             │
#  │                                                                         │
#  │ HOW to fix:                                                             │
#  │  - Try a more powerful model (LogReg, SVM, RandomForest)               │
#  │  - Increase max_features or add bigrams (ngram_range=(1,2))            │
#  │  - Reduce stopword aggressiveness                                       │
#  │  - Add more training samples                                            │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  GOOD FIT
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ Train score: HIGH   Test score: CLOSE to train (gap ≤ 0.10)            │
#  │ Model generalises well — this is the target zone                       │
#  └─────────────────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import learning_curve

from src.config import METRICS_PATH, FIGURES_PATH
from src.utils import ensure_dir


def evaluate_model(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    """Compute and return all four classification metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)

    # Accuracy: overall correctness — good baseline but can hide class imbalance
    accuracy = round(accuracy_score(y_test, y_pred), 4)

    # Precision: of all predicted positives, how many were correct
    precision = round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4)

    # Recall: of all actual positives, how many did the model find
    recall = round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4)

    # F1: harmonic mean of precision and recall — best single metric for comparison
    f1 = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compute_train_test_scores(
    pipeline: Pipeline,
    X_train: pd.Series, y_train: pd.Series,
    X_test: pd.Series,  y_test: pd.Series,
) -> dict:
    """Return train and test accuracy scores for over/underfitting diagnosis.

    The gap between these two numbers is the primary signal:
    - Small gap, both high  → good generalisation
    - Large gap             → overfitting (model memorised training data)
    - Both low              → underfitting (model not learning enough)
    """
    return {
        "train_score": round(pipeline.score(X_train, y_train), 4),
        "test_score":  round(pipeline.score(X_test,  y_test),  4),
    }


def diagnose_fit(train_score: float, test_score: float, threshold: float = 0.10) -> str:
    """Classify model fit as 'overfit', 'underfit', or 'good_fit'.

    Decision rules:
    1. Both scores < 0.70           → underfit  (model too simple)
    2. train - test > threshold     → overfit   (model memorised training data)
    3. Otherwise                    → good_fit  (generalises well)

    The threshold of 0.10 is a practical heuristic — a 10-point gap between
    train and test accuracy is a clear sign the model is not generalising.
    """
    gap = train_score - test_score

    # Both scores low → model hasn't learned the task at all
    if train_score < 0.70 and test_score < 0.70:
        return "underfit"

    # High train but much lower test → model memorised, not generalised
    if gap > threshold:
        return "overfit"

    # Scores are close and reasonably high → model generalises well
    return "good_fit"


def print_fit_diagnosis(model_name: str, train_score: float,
                        test_score: float, diagnosis: str) -> None:
    """Print a detailed train vs test comparison with explanation and fix advice."""
    gap = train_score - test_score

    print(f"\n  {'─' * 46}")
    print(f"  Fit Diagnosis — {model_name}")
    print(f"  {'─' * 46}")
    print(f"  Train score : {train_score:.4f}  (performance on data the model SAW)")
    print(f"  Test score  : {test_score:.4f}  (performance on data the model NEVER saw)")
    print(f"  Gap         : {gap:+.4f}")
    print()

    if diagnosis == "overfit":
        # Overfit: model learned training data too well, fails on new data
        print("  ⚠  OVERFIT detected")
        print("  The model performs much better on training data than test data.")
        print("  It has memorised patterns specific to training examples")
        print("  rather than learning generalisable rules.")
        print()
        print("  Likely causes:")
        print("  - Model too complex (e.g. Decision Tree with no depth limit)")
        print("  - Too many TF-IDF features relative to training samples")
        print("  - Training set too small")
        print()
        print("  Suggested fixes:")
        print("  - Add regularisation (reduce C in LogReg/SVM)")
        print("  - Reduce max_features in TF-IDF config")
        print("  - Set max_depth on DecisionTree or min_samples_leaf on RF")
        print("  - Collect more training data")

    elif diagnosis == "underfit":
        # Underfit: model is too simple to capture the signal in the data
        print("  ⚠  UNDERFIT detected")
        print("  Both train and test scores are low — the model is not")
        print("  learning enough from the training data.")
        print()
        print("  Likely causes:")
        print("  - Model too simple for the task")
        print("  - TF-IDF vocabulary too small (max_features too low)")
        print("  - Over-aggressive preprocessing removed meaningful tokens")
        print()
        print("  Suggested fixes:")
        print("  - Try a more powerful model (LogReg, SVM, RandomForest)")
        print("  - Increase max_features or add bigrams (ngram_range=(1,2))")
        print("  - Reduce stopword aggressiveness in preprocessing")
        print("  - Add more training samples")

    else:
        # Good fit: train and test scores are close — model generalises well
        print("  ✓  GOOD FIT")
        print("  Train and test scores are close — the model generalises well.")
        print("  It has learned patterns that transfer to unseen data.")

    print(f"  {'─' * 46}")


def print_metrics(metrics: dict, model_name: str) -> None:
    """Print a clearly formatted evaluation report for a single model."""
    print(f"\n{'─' * 50}")
    print(f"  Model : {model_name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  "
          f"→ {metrics['accuracy'] * 100:.1f}% of predictions correct")
    print(f"  Precision : {metrics['precision']:.4f}  "
          f"→ reliability of positive predictions")
    print(f"  Recall    : {metrics['recall']:.4f}  "
          f"→ coverage of actual positive cases")
    print(f"  F1 Score  : {metrics['f1']:.4f}  "
          f"→ harmonic mean of precision & recall")
    print(f"{'─' * 50}")

    # Print detailed fit diagnosis if scores are available
    if "train_score" in metrics and "test_score" in metrics:
        print_fit_diagnosis(
            model_name,
            metrics["train_score"],
            metrics["test_score"],
            metrics.get("diagnosis", "n/a"),
        )


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
    """Save a seaborn confusion matrix heatmap to reports/figures/.

    How to read it:
    - Rows = actual (true) labels
    - Columns = predicted labels
    - Diagonal cells = correct predictions (want these high)
    - Off-diagonal cells = errors (want these low)
    """
    ensure_dir(FIGURES_PATH)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = pipeline.classes_ if hasattr(pipeline, "classes_") else "auto"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
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
    """Save a learning curve (train vs CV score vs training size) to reports/figures/.

    How to read it:
    - Both lines converging at a high score → good fit
    - Large persistent gap between lines    → overfit
    - Both lines flat and low               → underfit
    - CV score still rising at max size     → more data would help
    """
    ensure_dir(FIGURES_PATH)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=3, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1,
    )
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train score")
    plt.plot(train_sizes, val_scores.mean(axis=1),   label="CV score")
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.1)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.1)
    plt.title(f"Learning Curve — {model_name}")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
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
