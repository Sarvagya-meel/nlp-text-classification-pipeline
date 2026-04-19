# run_experiments.py
# Runnable script for L1/L2 regularization, cross-validation, and hyperparameter tuning.
# Usage: python3 run_experiments.py --data data/raw/dataset.csv

import argparse
from sklearn.model_selection import train_test_split

from src.utils import ensure_nltk_resources
from src.config import TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_raw_data, validate_columns
from src.preprocessing import remove_missing, preprocess_series
from src.experiments import (
    compare_l1_l2, plot_l1_l2_comparison,
    run_cross_validation, plot_cv_results,
    run_hyperparameter_tuning, save_experiment_results,
)


def main(csv_path: str) -> None:
    """Run all three experiments on the dataset."""
    ensure_nltk_resources()

    # ── Load & preprocess ────────────────────────────────────────────────────
    print("Loading and preprocessing data...")
    df = load_raw_data(csv_path)
    validate_columns(df, [TEXT_COLUMN, LABEL_COLUMN])
    df = remove_missing(df, TEXT_COLUMN)
    df[TEXT_COLUMN] = preprocess_series(df[TEXT_COLUMN])
    print(f"  {len(df)} samples ready.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN], df[LABEL_COLUMN],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN],
    )

    # ── Experiment 1: L1 vs L2 Regularization ───────────────────────────────
    print("=" * 60)
    print("  EXPERIMENT 1 — L1 vs L2 Regularization")
    print("  Comparing LogReg with L1 (sparse) and L2 (shrink) penalties")
    print("  across C values from 0.001 (strong) to 100 (weak)")
    print("=" * 60)
    l1_l2_results = compare_l1_l2(X_train, y_train, X_test, y_test)
    plot_l1_l2_comparison(l1_l2_results)
    save_experiment_results(l1_l2_results, "l1_l2_results")

    print("\n  Key insight:")
    print("  - L1 tends to zero out irrelevant TF-IDF features (sparse model)")
    print("  - L2 keeps all features but shrinks them (stable on correlated features)")
    print("  - Best C is where test score peaks — too small = underfit, too large = overfit")

    # ── Experiment 2: Cross-Validation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2 — 5-Fold Cross-Validation")
    print("  More reliable than a single train/test split.")
    print("  Each model trained and evaluated 5 times on different data splits.")
    print("=" * 60)
    cv_results = run_cross_validation(df[TEXT_COLUMN], df[LABEL_COLUMN], cv=5)
    plot_cv_results(cv_results)
    save_experiment_results(cv_results, "cross_validation_results")

    print("\n  Key insight:")
    print("  - Mean score = expected real-world performance")
    print("  - Std dev > 0.05 = model is unstable (sensitive to data split)")
    print("  - Decision Tree typically shows high std — confirms overfitting tendency")

    # ── Experiment 3: Hyperparameter Tuning ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3 — Hyperparameter Tuning (GridSearchCV)")
    print("  Searching over TF-IDF params + LogReg C and penalty.")
    print("  24 combinations × 5 folds = 120 model fits.")
    print("=" * 60)
    tuning_results = run_hyperparameter_tuning(X_train, y_train, X_test, y_test)
    save_experiment_results(tuning_results, "hyperparameter_tuning_results")

    print("\n  Key insight:")
    print("  - GridSearchCV uses CV internally — best params chosen without touching test set")
    print("  - This prevents data leakage from hyperparameter selection")
    print("  - The best params tell you which TF-IDF vocab size and regularization work best")

    print("\n" + "=" * 60)
    print("  All experiments complete.")
    print("  Plots  → reports/figures/")
    print("  Results → reports/metrics/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Experiments: L1/L2, CV, Tuning")
    parser.add_argument("--data", required=True, help="Path to raw CSV file")
    args = parser.parse_args()
    main(args.data)
