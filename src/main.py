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
    evaluate_model, compute_train_test_scores, diagnose_fit,
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

        plot_confusion_matrix(fitted, X_test, y_test, name)
        plot_learning_curve(fitted, X_train, y_train, name)

        all_metrics[name] = metrics

    plot_metrics_comparison(all_metrics)

    # 6. Summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'Fit':>10}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(
            f"{name:<22} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['diagnosis']:>10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Text Classification Pipeline")
    parser.add_argument("--data", required=True, help="Path to raw CSV file")
    args = parser.parse_args()
    run_pipeline(args.data)
