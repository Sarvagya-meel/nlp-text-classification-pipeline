# Project Structure & Coding Rules

## Folder Structure

```
nlp-text-classification-pipeline/
├── data/
│   ├── raw/                  # Original unprocessed datasets
│   └── processed/            # Cleaned/tokenized data ready for training
├── models/                   # Saved model and vectorizer artifacts (.joblib)
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Text cleaning, tokenization, stopwords, stemming
│   ├── features.py           # TF-IDF vectorizer fitting and transformation
│   ├── train.py              # Classifier training and model persistence
│   ├── evaluate.py           # Metrics: accuracy, precision, recall, F1, confusion matrix
│   └── predict.py            # Inference on new text input
├── scripts/
│   ├── run_train.py          # CLI entry point for training
│   └── run_predict.py        # CLI entry point for inference
├── tests/
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_train.py
│   └── test_evaluate.py
├── notebooks/                # Optional EDA / experimentation notebooks
├── requirements.txt
└── README.md
```

## Module Responsibilities

- `preprocess.py` — pure text transformation functions, no I/O
- `features.py` — TF-IDF fitting/transforming; returns sklearn-compatible objects
- `train.py` — accepts vectorized features, trains classifiers, saves artifacts
- `evaluate.py` — accepts predictions and ground truth, returns metrics dict
- `predict.py` — loads saved model + vectorizer, runs full inference on raw text

## Coding Rules

1. Each module has a single responsibility — do not mix preprocessing with training logic
2. Functions must be pure where possible (same input → same output, no side effects)
3. All file paths passed as parameters, never hardcoded inside functions
4. Classifiers are selected by name string (e.g., `"logreg"`, `"svm"`) mapped to instances in a registry dict
5. Tests use `pytest`; each test file mirrors its corresponding `src/` module
6. No Jupyter notebooks in `src/` — notebooks go in `notebooks/` only
7. Keep functions short — if a function exceeds ~30 lines, consider splitting it
