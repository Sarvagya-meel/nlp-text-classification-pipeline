# src/experiments.py
# Advanced experiments: L1/L2 regularization, cross-validation, hyperparameter tuning.
#
# ─────────────────────────────────────────────────────────────────────────────
# L1 vs L2 REGULARIZATION
# ─────────────────────────────────────────────────────────────────────────────
# Regularization adds a penalty to the loss function to prevent overfitting
# by discouraging large weights.
#
# L1 (Lasso) — penalty = sum of |weights|
#   - Pushes some weights exactly to zero → automatic feature selection
#   - Produces sparse models (only important features survive)
#   - Good when you suspect many features are irrelevant
#   - In LogReg: penalty='l1', solver='liblinear'
#
# L2 (Ridge) — penalty = sum of weights²
#   - Shrinks all weights toward zero but never exactly to zero
#   - Distributes weight across correlated features
#   - More stable when features are correlated (common in TF-IDF)
#   - In LogReg: penalty='l2' (default)
#
# C parameter: inverse of regularization strength
#   - Small C → strong regularization → simpler model → less overfit
#   - Large C → weak regularization → complex model → may overfit
#
# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
# A single train/test split gives one score that depends on which samples
# happened to land in the test set. Cross-validation (k-fold) splits the data
# into k folds, trains on k-1 folds, tests on the remaining fold, and repeats
# k times. The final score is the mean ± std across all folds.
#
# Why it matters:
#   - More reliable estimate of real-world performance
#   - Detects high variance (large std = unstable model)
#   - Uses all data for both training and evaluation
#
# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters are settings chosen BEFORE training (not learned from data).
# GridSearchCV exhaustively tries all combinations and picks the best one
# using cross-validation — so the selection is unbiased.
#
# Key hyperparameters for this pipeline:
#   TF-IDF: max_features, ngram_range
#   LogReg: C (regularization strength), penalty (L1 vs L2)
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

from src.config import FIGURES_PATH, METRICS_PATH
from src.utils import ensure_dir


# ─────────────────────────────────────────────────────────────────────────────
# L1 vs L2 REGULARIZATION
# ─────────────────────────────────────────────────────────────────────────────

def compare_l1_l2(
    X_train: pd.Series, y_train: pd.Series,
    X_test: pd.Series,  y_test: pd.Series,
) -> dict:
    """Train LogReg with L1 and L2 penalties across C values and compare scores.

    Returns a dict with results for plotting and reporting.
    """
    # C values to test: from strong regularization (0.001) to weak (100)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    results = {"C_values": C_values, "l1": [], "l2": []}

    for penalty in ["l1", "l2"]:
        for C in C_values:
            # Use l1_ratio to specify penalty type (sklearn 1.8+ API)
            # l1_ratio=1 → L1 (sparse), l1_ratio=0 → L2 (shrink all weights)
            l1_ratio = 1.0 if penalty == "l1" else 0.0
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                ("clf", LogisticRegression(
                    l1_ratio=l1_ratio,
                    C=C,
                    solver="saga",      # supports both L1 and L2 via l1_ratio
                    max_iter=2000,
                )),
            ])
            pipeline.fit(X_train, y_train)
            score = round(pipeline.score(X_test, y_test), 4)
            results[penalty].append(score)

            print(f"  LogReg penalty={penalty:<2}  C={C:<6}  test_score={score:.4f}")

    return results


def plot_l1_l2_comparison(results: dict) -> None:
    """Save a line chart comparing L1 vs L2 test scores across C values."""
    ensure_dir(FIGURES_PATH)
    C_values = results["C_values"]
    x = range(len(C_values))

    plt.figure(figsize=(8, 5))
    plt.plot(x, results["l1"], marker="o", label="L1 (Lasso) — sparse weights")
    plt.plot(x, results["l2"], marker="s", label="L2 (Ridge) — shrunk weights")
    plt.xticks(x, [str(c) for c in C_values])
    plt.xlabel("C (inverse regularization strength →  larger C = less regularization)")
    plt.ylabel("Test Accuracy")
    plt.title("L1 vs L2 Regularization — Effect of C on Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{FIGURES_PATH}l1_l2_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"\n  Plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    X: pd.Series, y: pd.Series, cv: int = 5
) -> dict:
    """Run k-fold cross-validation on all key models and return mean ± std scores.

    Uses StratifiedKFold to preserve class distribution in each fold.
    """
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    models = {
        "naive_bayes":         MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm":                 LinearSVC(max_iter=1000),
        "decision_tree":       DecisionTreeClassifier(),
        "random_forest":       RandomForestClassifier(n_estimators=100),
    }

    # StratifiedKFold ensures each fold has the same class ratio as the full dataset
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = {}

    print(f"\n  {cv}-Fold Stratified Cross-Validation Results")
    print(f"  {'─' * 55}")
    print(f"  {'Model':<24} {'Mean Acc':>9} {'Std':>8}  {'Interpretation'}")
    print(f"  {'─' * 55}")

    for name, clf in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ("clf", clf),
        ])
        # cross_val_score trains and evaluates on each fold automatically
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        mean_score = round(scores.mean(), 4)
        std_score  = round(scores.std(), 4)

        # High std = unstable model (sensitive to which data it sees)
        interpretation = "stable" if std_score < 0.05 else "unstable — high variance"

        cv_results[name] = {"mean": mean_score, "std": std_score, "scores": scores.tolist()}
        print(f"  {name:<24} {mean_score:>9.4f} {std_score:>8.4f}  {interpretation}")

    print(f"  {'─' * 55}")
    return cv_results


def plot_cv_results(cv_results: dict) -> None:
    """Save a bar chart of cross-validation mean scores with std error bars."""
    ensure_dir(FIGURES_PATH)
    models = list(cv_results.keys())
    means  = [cv_results[m]["mean"] for m in models]
    stds   = [cv_results[m]["std"]  for m in models]

    x = np.arange(len(models))
    plt.figure(figsize=(9, 5))
    bars = plt.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title("5-Fold Cross-Validation — Mean Accuracy ± Std Dev\n"
              "(Error bars show variance — smaller = more stable)")
    plt.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="0.70 threshold")
    plt.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}cross_validation_results.png"
    plt.savefig(path)
    plt.close()
    print(f"  Plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_tuning(
    X_train: pd.Series, y_train: pd.Series,
    X_test: pd.Series,  y_test: pd.Series,
) -> dict:
    """Run GridSearchCV over TF-IDF + LogReg hyperparameters.

    Searches over:
      - TF-IDF: max_features, ngram_range
      - LogReg: C (regularization strength), penalty (L1 vs L2)

    Uses 5-fold CV internally so the best params are chosen without
    touching the test set — no data leakage.
    """
def run_hyperparameter_tuning(
    X_train: pd.Series, y_train: pd.Series,
    X_test: pd.Series,  y_test: pd.Series,
) -> dict:
    """Run GridSearchCV over TF-IDF + LogReg hyperparameters.

    Searches over:
      - TF-IDF: max_features, ngram_range
      - LogReg: C (regularization strength), l1_ratio (0.0=L2, 1.0=L1)

    Uses 5-fold CV internally so the best params are chosen without
    touching the test set — no data leakage.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   LogisticRegression(solver="saga", max_iter=2000)),
    ])

    # Parameter grid: all combinations will be tried
    # Total: 2 × 2 × 3 × 2 = 24 combinations × 5 folds = 120 fits
    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range":  [(1, 1), (1, 2)],
        "clf__C":              [0.1, 1, 10],
        "clf__l1_ratio":       [0.0, 1.0],   # 0.0 = L2 (shrink), 1.0 = L1 (sparse)
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_params  = grid_search.best_params_
    best_cv_score = round(grid_search.best_score_, 4)
    test_score   = round(grid_search.best_estimator_.score(X_test, y_test), 4)

    print(f"\n  Best Parameters Found:")
    for k, v in best_params.items():
        print(f"    {k:<30} = {v}")
    print(f"\n  Best CV score  : {best_cv_score:.4f}")
    print(f"  Test score     : {test_score:.4f}")

    return {
        "best_params":   best_params,
        "best_cv_score": best_cv_score,
        "test_score":    test_score,
    }


def save_experiment_results(results: dict, filename: str) -> None:
    """Save experiment results as JSON to reports/metrics/."""
    ensure_dir(METRICS_PATH)
    path = f"{METRICS_PATH}{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {path}")
