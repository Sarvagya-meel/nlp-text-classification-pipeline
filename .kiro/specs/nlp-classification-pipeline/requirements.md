# Requirements Document: NLP Classification Pipeline

## Introduction

This document defines the functional and non-functional requirements for the NLP Classification Pipeline feature. The pipeline ingests raw text data from CSV files, preprocesses and vectorizes the text, trains five classifiers under identical conditions, evaluates and persists results, and supports inference on new input. All requirements are derived from the design document.

---

## Requirements

### 1. Data Loading

#### 1.1 Load CSV Data

**User Story**: As a data scientist, I want to load a raw text dataset from a CSV file so that I can feed it into the classification pipeline.

**Acceptance Criteria**:
- [ ] `load_raw_data(filepath)` reads a CSV file and returns a `pd.DataFrame`
- [ ] The function accepts any valid filesystem path string
- [ ] The returned DataFrame preserves all columns from the source CSV
- [ ] An `IOError` or `FileNotFoundError` is raised if the file does not exist

#### 1.2 Validate Required Columns

**User Story**: As a developer, I want the pipeline to fail fast with a clear error if required columns are missing so that I can fix configuration issues immediately.

**Acceptance Criteria**:
- [ ] `validate_columns(df, required)` raises `ValueError` if any column in `required` is absent from `df`
- [ ] The error message lists all missing column names
- [ ] No exception is raised when all required columns are present

---

### 2. Preprocessing

#### 2.1 Handle Missing Text Values

**User Story**: As a data scientist, I want rows with null or empty text to be removed before processing so that downstream steps receive only valid input.

**Acceptance Criteria**:
- [ ] `remove_missing(df, col)` drops all rows where `col` is `NaN` or an empty string after stripping whitespace
- [ ] The returned DataFrame has fewer or equal rows than the input
- [ ] A `ValueError` is raised if the resulting DataFrame is empty
- [ ] The original DataFrame is not mutated

#### 2.2 Clean and Normalize Text

**User Story**: As a data scientist, I want raw text to be lowercased, stripped of HTML and punctuation, and whitespace-normalized so that the vectorizer receives consistent input.

**Acceptance Criteria**:
- [ ] `clean_text(text)` returns a lowercase string
- [ ] HTML tags are removed from the output
- [ ] Punctuation characters are removed
- [ ] Multiple consecutive whitespace characters are collapsed to a single space
- [ ] Leading and trailing whitespace is stripped
- [ ] Applying `clean_text` twice to the same input produces the same result (idempotent)

#### 2.3 Tokenization

**User Story**: As a data scientist, I want text to be split into tokens using NLTK so that stopword removal and morphological normalization can be applied.

**Acceptance Criteria**:
- [ ] `tokenize(text)` returns a `list[str]` using `nltk.word_tokenize`
- [ ] An empty string input returns an empty list
- [ ] Tokens are lowercase strings (applied after `clean_text`)

#### 2.4 Stopword Removal

**User Story**: As a data scientist, I want common English stopwords removed from token lists so that the TF-IDF vocabulary focuses on meaningful terms.

**Acceptance Criteria**:
- [ ] `remove_stopwords(tokens)` returns a list with no tokens present in the NLTK English stopword set
- [ ] Non-stopword tokens are preserved in their original order
- [ ] The function accepts an optional `lang` parameter defaulting to `"english"`

#### 2.5 Stemming and Lemmatization

**User Story**: As a data scientist, I want to apply either stemming or lemmatization to tokens so that morphological variants of the same word are unified.

**Acceptance Criteria**:
- [ ] `stem_tokens(tokens)` applies `PorterStemmer` to each token and returns the stemmed list
- [ ] `lemmatize_tokens(tokens)` applies `WordNetLemmatizer` to each token and returns the lemmatized list
- [ ] `preprocess_series` accepts `use_stemming` and `use_lemmatization` boolean flags
- [ ] `use_stemming=True` and `use_lemmatization=True` simultaneously is not permitted; the function raises `ValueError`
- [ ] When both flags are `False`, tokens are returned without morphological normalization

#### 2.6 Full Preprocessing Series

**User Story**: As a developer, I want a single function to apply the complete preprocessing pipeline to a text Series so that I can call it in one line.

**Acceptance Criteria**:
- [ ] `preprocess_series(series)` returns a `pd.Series` of equal length to the input
- [ ] Every element in the output is a non-null string
- [ ] The function applies cleaning, tokenization, stopword removal, and optional stemming/lemmatization in sequence
- [ ] Applying `preprocess_series` twice to the same input produces the same output (idempotent)

---

### 3. Feature Extraction

#### 3.1 TF-IDF Vectorization with N-grams

**User Story**: As a data scientist, I want TF-IDF features with unigrams and bigrams so that both single-word and two-word patterns are captured.

**Acceptance Criteria**:
- [ ] `build_tfidf(ngram_range, max_features, stop_words)` returns a `TfidfVectorizer` configured with the given parameters
- [ ] The default `ngram_range` is `(1, 2)` as defined in `config.py`
- [ ] The fitted vectorizer vocabulary contains at least one bigram (token pair)
- [ ] `max_features` limits the vocabulary size when set

#### 3.2 Fit on Training Data Only

**User Story**: As a data scientist, I want the vectorizer to be fitted exclusively on training data so that there is no data leakage from the test set.

**Acceptance Criteria**:
- [ ] `fit_transform_tfidf(vectorizer, X_train)` fits the vectorizer on `X_train` and returns the transformed sparse matrix
- [ ] `transform_tfidf(vectorizer, X)` transforms `X` without modifying the vectorizer's vocabulary
- [ ] Calling `transform_tfidf` on test data does not change `vectorizer.vocabulary_`
- [ ] When using `sklearn.Pipeline`, the vectorizer step is fitted only during `pipeline.fit(X_train, y_train)`

---

### 4. Model Training

#### 4.1 Build Five sklearn Pipelines

**User Story**: As a data scientist, I want one `sklearn.Pipeline` per classifier so that TF-IDF and classification are encapsulated together and data leakage is prevented.

**Acceptance Criteria**:
- [ ] `build_pipelines(tfidf_params)` returns a `dict` with exactly 5 entries
- [ ] Keys are: `"naive_bayes"`, `"logistic_regression"`, `"svm"`, `"decision_tree"`, `"random_forest"`
- [ ] Each value is an unfitted `sklearn.Pipeline` with steps named `"tfidf"` and `"clf"`
- [ ] The `"tfidf"` step is a `TfidfVectorizer`; the `"clf"` step is the corresponding classifier
- [ ] No pipeline is fitted at construction time

#### 4.2 Train All Models on Identical Data Splits

**User Story**: As a data scientist, I want all five models trained on the same train/test split so that evaluation results are directly comparable.

**Acceptance Criteria**:
- [ ] `train_model(pipeline, X_train, y_train)` calls `pipeline.fit(X_train, y_train)` and returns the fitted pipeline
- [ ] All five pipelines receive the same `X_train` and `y_train` objects
- [ ] The train/test split uses `stratify=y` to preserve class distribution
- [ ] `RANDOM_STATE` from `config.py` is used for reproducibility

#### 4.3 Save Trained Models

**User Story**: As a developer, I want trained models persisted to disk so that they can be loaded for inference without retraining.

**Acceptance Criteria**:
- [ ] `save_model(pipeline, model_name)` writes a `.joblib` file to `models/{model_name}.joblib`
- [ ] The file is readable by `joblib.load` and produces a valid `Pipeline` object
- [ ] The `models/` directory is created if it does not exist

---

### 5. Evaluation

#### 5.1 Compute Per-Model Metrics

**User Story**: As a data scientist, I want accuracy, precision, recall, and F1 reported for each model so that I can compare their performance.

**Acceptance Criteria**:
- [ ] `evaluate_model(pipeline, X_test, y_test)` returns a `dict` with keys `"accuracy"`, `"precision"`, `"recall"`, `"f1"`
- [ ] All metric values are floats in `[0.0, 1.0]`
- [ ] Precision, recall, and F1 are macro-averaged across classes
- [ ] The fitted pipeline is not mutated by this function

#### 5.2 Overfitting and Underfitting Detection

**User Story**: As a data scientist, I want train and test scores compared per model so that I can identify overfitting or underfitting.

**Acceptance Criteria**:
- [ ] `compute_train_test_scores(pipeline, X_train, y_train, X_test, y_test)` returns a `dict` with keys `"train_score"` and `"test_score"`
- [ ] Both scores are floats in `[0.0, 1.0]`
- [ ] A gap of `train_score - test_score > 0.10` is flagged as potential overfitting in the saved metrics
- [ ] Both scores below `0.70` are flagged as potential underfitting

#### 5.3 Learning Curve Plots

**User Story**: As a data scientist, I want learning curve plots per model so that I can visually diagnose over/underfitting across training set sizes.

**Acceptance Criteria**:
- [ ] `plot_learning_curve(pipeline, X, y, model_name)` saves a PNG to `reports/figures/{model_name}_learning_curve.png`
- [ ] The plot shows training score and cross-validation score as a function of training set size
- [ ] The file is created even if the directory did not previously exist

#### 5.4 Confusion Matrix Plots

**User Story**: As a data scientist, I want a confusion matrix heatmap per model so that I can see per-class prediction errors.

**Acceptance Criteria**:
- [ ] `plot_confusion_matrix(pipeline, X_test, y_test, model_name)` saves a PNG to `reports/figures/{model_name}_confusion_matrix.png`
- [ ] The heatmap uses seaborn styling
- [ ] Class labels are shown on both axes

#### 5.5 Cross-Model Comparison Chart

**User Story**: As a data scientist, I want a single chart comparing all five models across all four metrics so that I can identify the best-performing model at a glance.

**Acceptance Criteria**:
- [ ] `plot_metrics_comparison(all_metrics)` saves a grouped bar chart to `reports/figures/metrics_comparison.png`
- [ ] All five models and all four metrics (accuracy, precision, recall, F1) are represented
- [ ] The chart is saved even if some models have lower scores

#### 5.6 Persist Metrics to Disk

**User Story**: As a developer, I want metric results saved as JSON files so that they can be reviewed and compared programmatically.

**Acceptance Criteria**:
- [ ] `save_metrics(metrics, model_name)` writes a JSON file to `reports/metrics/{model_name}_metrics.json`
- [ ] The JSON contains all keys from the metrics dict including `train_score`, `test_score`, and the four evaluation metrics
- [ ] The `reports/metrics/` directory is created if it does not exist

---

### 6. Inference

#### 6.1 Load Saved Model

**User Story**: As a developer, I want to load a previously saved model by name so that I can run predictions without retraining.

**Acceptance Criteria**:
- [ ] `load_model(model_name)` loads and returns the pipeline from `models/{model_name}.joblib`
- [ ] A `FileNotFoundError` is raised with the expected path if the file does not exist
- [ ] The returned object is a fitted `sklearn.Pipeline`

#### 6.2 Predict on New Text Input

**User Story**: As a developer, I want to pass a raw text string to the inference module and receive a predicted label and confidence score so that the model can be used in production.

**Acceptance Criteria**:
- [ ] `predict(text, model_name)` returns a tuple `(label: str, confidence: float)`
- [ ] `confidence` is in `[0.0, 1.0]`
- [ ] The input text is passed through the same preprocessing steps as training data before prediction
- [ ] For `LinearSVC`, confidence is derived from the calibrated decision function score
- [ ] A `ValueError` is raised if `text` is empty or whitespace-only

---

### 7. Configuration and Structure

#### 7.1 Centralized Configuration

**User Story**: As a developer, I want all constants defined in `config.py` so that I can change paths and hyperparameters in one place.

**Acceptance Criteria**:
- [ ] `src/config.py` defines: `DATA_RAW_PATH`, `DATA_PROCESSED_PATH`, `MODELS_PATH`, `METRICS_PATH`, `FIGURES_PATH`, `TEXT_COLUMN`, `LABEL_COLUMN`, `TEST_SIZE`, `RANDOM_STATE`, `TFIDF_MAX_FEATURES`, `TFIDF_NGRAM_RANGE`, `TFIDF_STOP_WORDS`
- [ ] No other module contains hardcoded path strings or magic numbers
- [ ] All modules import constants explicitly from `src.config`

#### 7.2 Module Single Responsibility

**User Story**: As a developer, I want each source module to have exactly one responsibility so that the codebase is easy to navigate and extend.

**Acceptance Criteria**:
- [ ] `data_loader.py` contains only data loading and column validation logic
- [ ] `preprocessing.py` contains only text cleaning and normalization logic
- [ ] `features.py` contains only TF-IDF vectorization logic
- [ ] `train.py` contains only pipeline construction and model training logic
- [ ] `evaluate.py` contains only metric computation and plot generation logic
- [ ] `inference.py` contains only model loading and prediction logic
- [ ] `utils.py` contains only shared helper utilities

#### 7.3 NLTK Resource Availability

**User Story**: As a developer, I want NLTK resources auto-downloaded at pipeline startup so that the pipeline runs without manual setup steps.

**Acceptance Criteria**:
- [ ] `utils.ensure_nltk_resources()` downloads `stopwords`, `wordnet`, and `punkt` if not already present
- [ ] The function is called at the entry point before any preprocessing begins
- [ ] No `LookupError` is raised during preprocessing when this function has been called
