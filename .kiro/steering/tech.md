---
inclusion: always
---

# Technical Guidelines

## Language & Runtime
- Python 3.10+
- All scripts must be runnable from the project root

## Core Dependencies
- `pandas`, `numpy` — data handling
- `scikit-learn` — ML pipelines, vectorization, models, metrics
- `nltk` — tokenization, stopwords, stemming/lemmatization
- `joblib` — model serialization
- `matplotlib`, `seaborn` — visualization

## Optional Dependencies
- `spaCy` — advanced NLP (named entity recognition, lemmatization)
- `FastAPI` — serving predictions via API layer

## Feature Extraction
- Default: TF-IDF via `sklearn.feature_extraction.text.TfidfVectorizer`
- Always include unigrams and bigrams: `ngram_range=(1, 2)`
- Fit vectorizer on training data only; transform test data separately
- Reuse fitted vectorizers — never refit on test or inference data

## Models to Compare
All classifiers should be trained and evaluated under the same conditions:
- `MultinomialNB` — Naive Bayes baseline
- `LogisticRegression` — strong linear baseline
- `LinearSVC` — SVM variant, efficient for text
- `DecisionTreeClassifier` — interpretable, prone to overfitting
- `RandomForestClassifier` — ensemble, more robust

## sklearn Pipelines
- Wrap `TfidfVectorizer` + classifier in a `Pipeline` for each model
- This prevents data leakage and simplifies training/inference code
- Example:
  ```python
  Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", LogisticRegression())])
  ```

## Evaluation
- Report all four metrics per model: accuracy, precision, recall, F1
- Use `classification_report` from sklearn for consistency
- Save metric outputs to `reports/metrics/`
- Save plots (confusion matrix, comparison charts) to `reports/figures/`

## Configuration
- All constants (paths, hyperparameters, column names) live in `src/config.py`
- Never hardcode paths or magic strings in other modules
- Import config values explicitly — avoid `import *`

## Code Style
- Function names: `snake_case`, descriptive (e.g., `load_raw_data`, `train_model`)
- One responsibility per function; keep functions under ~30 lines
- Add a one-line docstring to every public function
- Minimal inline comments — only explain non-obvious logic

## Engineering Practices
- Each module in `src/` has a single responsibility (see `structure.md`)
- Do not mix data loading, preprocessing, and training logic in one file
- Prefer returning values over side effects in functions
- Use `joblib.dump` / `joblib.load` for all model persistence to `models/`
