# src/main.py
# Entry point: runs the full training pipeline end-to-end.
# Usage: python -m src.main --data data/raw/dataset.csv

import argparse
from sklearn.model_selection import train_test_split

from src.utils import ensure_nltk_resources
from src.config import (
    TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE, TFIDF_PARAMS
)
from src.data_loader import load_raw_data, validate_columns
from src.preprocessing import remove_missing, preprocess_series
from src.train import build_pipelines, train_model, save_model
from src.evaluate import (
    evaluate_model, print_metrics, compute_train_test_scores, diagnose_fit,
    save_metrics, plot_confusion_matrix, plot_learning_curve,
    plot_metrics_comparison,
)


def run_pipeline(csv_path: str) -> None:
    """Execute the full NLP classification training pipeline."""
    ensure_nltk_resources()

    # 1. Load and validate
    print("Loading data...")
    df = load_raw_data(csv_path)
    validate_columns(df, [TEXT_COLUMN, LABEL_COLUMN])
    df = remove_missing(df, TEXT_COLUMN)
    print(f"  {len(df)} samples loaded.")

    # 2. Preprocess
    print("Preprocessing text...")
    df[TEXT_COLUMN] = preprocess_series(df[TEXT_COLUMN])

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN], df[LABEL_COLUMN],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN],
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 4. Build pipelines
    pipelines = build_pipelines(TFIDF_PARAMS)

    # 5. Train, evaluate, persist
    all_metrics = {}
    for name, pipeline in pipelines.items():
        print(f"\nTraining: {name}")
        fitted = train_model(pipeline, X_train, y_train)
        save_model(fitted, name)

        metrics = evaluate_model(fitted, X_test, y_test)
        scores = compute_train_test_scores(fitted, X_train, y_train, X_test, y_test)
        diagnosis = diagnose_fit(scores["train_score"], scores["test_score"])

        metrics.update(scores)
        metrics["diagnosis"] = diagnosis
        save_metrics(metrics, name)
        print_metrics(metrics, name)

        plot_confusion_matrix(fitted, X_test, y_test, name)
        plot_learning_curve(fitted, X_train, y_train, name)

        all_metrics[name] = metrics

    plot_metrics_comparison(all_metrics)

    # 6. Comparison table
    # All 5 models trained on identical TF-IDF features and the same train/test split
    # so scores are directly comparable across rows.
    print("\n")
    print("=" * 75)
    print("  MODEL COMPARISON — All models trained on identical TF-IDF features")
    print("=" * 75)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Diagnosis':>12}")
    print("  " + "-" * 71)

    # Find best F1 to highlight the winner
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1"])

    for name, m in all_metrics.items():
        marker = " ◀ best" if name == best_name else ""
        print(
            f"  {name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['diagnosis']:>12}{marker}"
        )

    print("  " + "-" * 71)
    print()
    print("  Metrics legend:")
    print("  Accuracy  — % of all predictions that were correct")
    print("  Precision — of predicted positives, how many were truly positive")
    print("  Recall    — of actual positives, how many did the model catch")
    print("  F1        — harmonic mean of Precision & Recall (best overall score)")
    print()
    print("  Diagnosis legend:")
    print("  good_fit  — train/test scores are close and high")
    print("  overfit   — train score >> test score (gap > 0.10)")
    print("  underfit  — both scores below 0.70")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Text Classification Pipeline")
    parser.add_argument("--data", required=True, help="Path to raw CSV file")
    args = parser.parse_args()
    run_pipeline(args.data)
