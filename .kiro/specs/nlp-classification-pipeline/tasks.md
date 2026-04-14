# Tasks: NLP Classification Pipeline

## Overview

Implementation tasks derived from the design and requirements documents. Tasks are ordered by dependency — each group can only begin once its prerequisites are complete. All code lives under `src/`; all constants are imported from `src/config.py`.

---

## Tasks

- [ ] 1. Project scaffold and configuration
  - [ ] 1.1 Create directory structure: `src/`, `data/raw/`, `data/processed/`, `models/`, `reports/metrics/`, `reports/figures/`, `tests/`, `notes/`
  - [ ] 1.2 Create `src/__init__.py` (empty)
  - [ ] 1.3 Create `src/config.py` with all constants: `DATA_RAW_PATH`, `DATA_PROCESSED_PATH`, `MODELS_PATH`, `METRICS_PATH`, `FIGURES_PATH`, `TEXT_COLUMN`, `LABEL_COLUMN`, `TEST_SIZE`, `RANDOM_STATE`, `TFIDF_MAX_FEATURES`, `TFIDF_NGRAM_RANGE`, `TFIDF_STOP_WORDS`, and a `TFIDF_PARAMS` dict
  - [ ] 1.4 Create `requirements.txt` listing: `pandas`, `numpy`, `scikit-learn`, `nltk`, `joblib`, `matplotlib`, `seaborn`, `scipy`, `hypothesis`

- [ ] 2. Utilities (`src/utils.py`)
  - [ ] 2.1 Implement `ensure_dir(path: str) -> None` — creates directory if absent
  - [ ] 2.2 Implement `get_timestamp() -> str` — returns UTC timestamp string
  - [ ] 2.3 Implement `log_info(message: str) -> None` — prints formatted log line
  - [ ] 2.4 Implement `ensure_nltk_resources() -> None` — downloads `stopwords`, `wordnet`, `punkt` if missing
  - [ ] 2.5 Write unit tests in `tests/test_utils.py` covering all four functions

- [ ] 3. Data loading (`src/data_loader.py`)
  - [ ] 3.1 Implement `load_raw_data(filepath: str) -> pd.DataFrame`
  - [ ] 3.2 Implement `validate_columns(df: pd.DataFrame, required: list[str]) -> None` — raises `ValueError` listing missing columns
  - [ ] 3.3 Write unit tests in `tests/test_data_loader.py`: valid CSV, missing file, missing columns, empty CSV

- [ ] 4. Preprocessing (`src/preprocessing.py`)
  - [ ] 4.1 Implement `clean_text(text: str) -> str` — lowercase, strip HTML, remove punctuation, collapse whitespace
  - [ ] 4.2 Implement `remove_missing(df: pd.DataFrame, col: str) -> pd.DataFrame` — drops null/empty rows, raises `ValueError` if result is empty
  - [ ] 4.3 Implement `tokenize(text: str) -> list[str]` using `nltk.word_tokenize`
  - [ ] 4.4 Implement `remove_stopwords(tokens: list[str], lang: str = "english") -> list[str]`
  - [ ] 4.5 Implement `stem_tokens(tokens: list[str]) -> list[str]` using `PorterStemmer`
  - [ ] 4.6 Implement `lemmatize_tokens(tokens: list[str]) -> list[str]` using `WordNetLemmatizer`
  - [ ] 4.7 Implement `preprocess_series(series: pd.Series, use_stemming: bool = False, use_lemmatization: bool = True) -> pd.Series` — raises `ValueError` if both flags are `True`
  - [ ] 4.8 Write unit tests in `tests/test_preprocessing.py`: HTML stripping, null removal, stopword removal, stemming output, lemmatization output, idempotency of `preprocess_series`, mutual exclusion of flags

- [ ] 5. Feature extraction (`src/features.py`)
  - [ ] 5.1 Implement `build_tfidf(ngram_range: tuple, max_features: int, stop_words: str | None) -> TfidfVectorizer`
  - [ ] 5.2 Implement `fit_transform_tfidf(vectorizer: TfidfVectorizer, X_train: pd.Series) -> csr_matrix`
  - [ ] 5.3 Implement `transform_tfidf(vectorizer: TfidfVectorizer, X: pd.Series) -> csr_matrix`
  - [ ] 5.4 Write unit tests in `tests/test_features.py`: vocabulary contains bigrams, `transform` does not change `vocabulary_`, `max_features` limits vocab size

- [ ] 6. Model training (`src/train.py`)
  - [ ] 6.1 Implement `build_pipelines(tfidf_params: dict) -> dict[str, Pipeline]` — returns exactly 5 named pipelines
  - [ ] 6.2 Implement `train_model(pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series) -> Pipeline`
  - [ ] 6.3 Implement `save_model(pipeline: Pipeline, model_name: str) -> None` — uses `joblib.dump` to `models/{model_name}.joblib`, calls `ensure_dir`
  - [ ] 6.4 Write unit tests in `tests/test_train.py`: pipeline count is 5, step names are `"tfidf"` and `"clf"`, correct classifier types, `save_model` creates file

- [ ] 7. Evaluation (`src/evaluate.py`)
  - [ ] 7.1 Implement `evaluate_model(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict` — returns `accuracy`, `precision`, `recall`, `f1`
  - [ ] 7.2 Implement `compute_train_test_scores(pipeline, X_train, y_train, X_test, y_test) -> dict` — returns `train_score`, `test_score`
  - [ ] 7.3 Implement `diagnose_fit(train_score: float, test_score: float, threshold: float = 0.10) -> str` — returns `"overfit"`, `"underfit"`, or `"good_fit"`
  - [ ] 7.4 Implement `save_metrics(metrics: dict, model_name: str) -> None` — writes JSON to `reports/metrics/{model_name}_metrics.json`
  - [ ] 7.5 Implement `plot_confusion_matrix(pipeline, X_test, y_test, model_name: str) -> None` — saves PNG to `reports/figures/{model_name}_confusion_matrix.png`
  - [ ] 7.6 Implement `plot_learning_curve(pipeline, X, y, model_name: str) -> None` — saves PNG to `reports/figures/{model_name}_learning_curve.png`
  - [ ] 7.7 Implement `plot_metrics_comparison(all_metrics: dict[str, dict]) -> None` — saves grouped bar chart to `reports/figures/metrics_comparison.png`
  - [ ] 7.8 Write unit tests in `tests/test_evaluate.py`: metric bounds `[0,1]`, JSON file created, PNG files created, `diagnose_fit` returns valid labels for boundary inputs

- [ ] 8. Inference (`src/inference.py`)
  - [ ] 8.1 Implement `load_model(model_name: str) -> Pipeline` — loads from `models/{model_name}.joblib`, raises `FileNotFoundError` if absent
  - [ ] 8.2 Implement `predict(text: str, model_name: str) -> tuple[str, float]` — preprocesses text, runs pipeline, returns `(label, confidence)`; handles `LinearSVC` via calibrated scores; raises `ValueError` for empty input
  - [ ] 8.3 Write unit tests in `tests/test_inference.py`: round-trip save/load/predict, `FileNotFoundError` on missing model, `ValueError` on empty text, confidence in `[0,1]`

- [ ] 9. Property-based tests
  - [ ] 9.1 Write `tests/test_properties.py` using `hypothesis`:
    - `preprocess_series` idempotency: `f(f(x)) == f(x)` for any string series
    - `evaluate_model` metric bounds: all values in `[0.0, 1.0]` for any valid split
    - `build_pipelines` always returns exactly 5 pipelines for any valid `tfidf_params`
    - `diagnose_fit` is total: returns one of `{"overfit", "underfit", "good_fit"}` for any `(train_score, test_score) ∈ [0,1]²`
    - `validate_columns` raises `ValueError` for any DataFrame missing at least one required column

- [ ] 10. End-to-end integration test
  - [ ] 10.1 Create `tests/test_integration.py` with a synthetic 100-row, 2-class CSV fixture
  - [ ] 10.2 Run full training pipeline and assert all 5 `.joblib` files exist in `models/`
  - [ ] 10.3 Assert all 5 JSON metric files exist in `reports/metrics/`
  - [ ] 10.4 Assert all expected PNG files exist in `reports/figures/` (confusion matrices, learning curves, comparison chart)
  - [ ] 10.5 Run `predict` on 3 sample texts for each model and assert valid `(label, confidence)` tuples are returned

- [ ] 11. Entry point script
  - [ ] 11.1 Create `src/main.py` (or `run_pipeline.py` at project root) that orchestrates the full training flow: load → preprocess → split → build pipelines → train → evaluate → save
  - [ ] 11.2 Call `utils.ensure_nltk_resources()` at startup
  - [ ] 11.3 Accept `--data` CLI argument for CSV path (default from `config.py`)
  - [ ] 11.4 Print a summary table of all model metrics to stdout after training completes
