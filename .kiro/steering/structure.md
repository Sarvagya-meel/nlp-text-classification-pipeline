---
inclusion: always
---

# Project Structure

## Root Layout

```
nlp-text-classification-pipeline/
├── data/
│   ├── raw/          # Original, unmodified datasets (CSV)
│   └── processed/    # Cleaned and normalized datasets
├── src/
│   ├── __init__.py
│   ├── config.py     # All constants: paths, hyperparameters, column names
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── models/           # Serialized models via joblib
├── reports/
│   ├── metrics/      # Evaluation outputs (text/JSON)
│   └── figures/      # Plots and visualizations (PNG)
├── notes/
│   ├── concepts.md
│   ├── formulas.md
│   ├── interview_qs.md
│   └── learning_log.md
├── tests/
├── requirements.txt
└── README.md
```

## Module Responsibilities

Each `src/` file has exactly one responsibility — do not mix concerns across files:

| File | Responsibility |
|---|---|
| `config.py` | All constants (paths, hyperparameters, column names). Never hardcode these elsewhere. |
| `data_loader.py` | Load raw CSV datasets into DataFrames |
| `preprocessing.py` | Clean and normalize text (tokenization, stopwords, stemming/lemmatization) |
| `features.py` | TF-IDF vectorization — fit on train, transform on test/inference |
| `train.py` | Build sklearn Pipelines and train all models |
| `evaluate.py` | Compute and save metrics (accuracy, precision, recall, F1) and plots |
| `inference.py` | Load a saved model and run predictions on new input |
| `utils.py` | Shared helper functions used across modules |

## Root-level Scripts

| File | Responsibility |
|---|---|
| `test_inference.py` | Standalone demo script — runs all 5 models on sample texts, prints results table |

## Architectural Rules

- **Single responsibility**: each module handles one concern only
- **No cross-contamination**: data loading, preprocessing, and training logic must never coexist in the same file
- **Config-driven**: all paths and magic strings must be imported from `config.py`
- **Pipeline pattern**: every model is wrapped in a `sklearn.Pipeline` (TfidfVectorizer + classifier) to prevent data leakage
- **Vectorizer discipline**: fit only on training data; reuse the fitted vectorizer for test and inference — never refit
- **Model persistence**: use `joblib.dump` / `joblib.load`; save to `models/`
- **Output paths**: metrics go to `reports/metrics/`, plots go to `reports/figures/`
- **Prefer return values over side effects** in functions; keep functions under ~30 lines
- **Tests**: always run via `python3 -m pytest tests/ -v` from project root — never `pytest` directly (ensures `src` imports resolve)
