# NLP Text Classification Pipeline

A modular, end-to-end text classification pipeline built with classical NLP and scikit-learn. Designed to be interview-ready, production-style, and easy to explain.

## Features

- **5 Classifiers**: Naive Bayes, Logistic Regression, SVM, Decision Tree, Random Forest
- **TF-IDF Vectorization**: Unigrams + bigrams with configurable vocabulary size
- **Full Preprocessing**: Text cleaning, tokenization, stopword removal, lemmatization
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1 + confusion matrices + learning curves
- **Overfitting/Underfitting Detection**: Train vs test score comparison with actionable fix suggestions
- **Model Persistence**: Save/load trained models via joblib
- **Inference API**: Predict on new text with confidence scores

## Project Structure

```
nlp-text-classification-pipeline/
├── data/
│   ├── raw/              # Original CSV datasets
│   └── processed/        # Cleaned datasets
├── src/
│   ├── config.py         # All constants (paths, hyperparameters)
│   ├── data_loader.py    # CSV loading and validation
│   ├── preprocessing.py  # Text cleaning, tokenization, lemmatization
│   ├── features.py       # TF-IDF vectorization
│   ├── train.py          # Pipeline construction and training
│   ├── evaluate.py       # Metrics, plots, over/underfitting diagnosis
│   ├── inference.py      # Load model and predict on new text
│   ├── utils.py          # Shared helpers
│   └── main.py           # Entry point for full training pipeline
├── models/               # Saved .joblib model files
├── reports/
│   ├── metrics/          # JSON metric files per model
│   └── figures/          # Confusion matrices, learning curves, comparison charts
├── notes/
│   ├── learning_log.md   # Concepts, examples, interview questions
│   └── concepts.md       # NLP/ML concept reference
├── tests/
│   └── test_inference.py # pytest unit tests for inference module
├── test_inference.py     # Standalone inference demo script
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd nlp-text-classification-pipeline
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download NLTK data (auto-downloaded on first run, or manually)

```bash
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Usage

### Train All Models

Run the full pipeline on the provided dataset:

```bash
python3 -m src.main --data data/raw/dataset.csv
```

This will:
- Load and preprocess the CSV
- Split into train/test (80/20)
- Train all 5 models on identical TF-IDF features
- Evaluate each model (accuracy, precision, recall, F1)
- Diagnose overfitting/underfitting
- Save models to `models/`
- Save metrics to `reports/metrics/`
- Save plots to `reports/figures/`
- Print a comparison table

### Test Inference

After training, test predictions on new text:

```bash
python3 test_inference.py
```

Or use the inference API directly in Python:

```python
from src.inference import predict

label, confidence = predict("This product is amazing!", "logistic_regression")
print(f"Predicted: {label} (confidence: {confidence:.2%})")
```

Available model names:
- `naive_bayes`
- `logistic_regression`
- `svm`
- `decision_tree`
- `random_forest`

## Configuration

All constants live in `src/config.py`:

```python
# Paths
DATA_RAW_PATH = "data/raw/"
MODELS_PATH = "models/"
METRICS_PATH = "reports/metrics/"
FIGURES_PATH = "reports/figures/"

# Column names
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF
TFIDF_NGRAM_RANGE = (1, 2)      # unigrams + bigrams
TFIDF_MAX_FEATURES = 10000      # vocabulary size cap
TFIDF_STOP_WORDS = "english"    # built-in stopword list
```

## Dataset Format

The pipeline expects a CSV with two columns:

| text | label |
|---|---|
| "This product is great!" | positive |
| "Terrible quality." | negative |

- `text`: raw text (will be preprocessed automatically)
- `label`: target class (any string label)

A sample dataset with 100 balanced positive/negative reviews is included at `data/raw/dataset.csv`.

## Outputs

After running the pipeline, you'll find:

### Models
- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`
- `models/svm.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`

### Metrics (JSON)
- `reports/metrics/{model_name}_metrics.json`

Example:
```json
{
  "accuracy": 0.95,
  "precision": 0.9545,
  "recall": 0.95,
  "f1": 0.9499,
  "train_score": 1.0,
  "test_score": 0.95,
  "diagnosis": "good_fit"
}
```

### Plots (PNG)
- `reports/figures/{model_name}_confusion_matrix.png`
- `reports/figures/{model_name}_learning_curve.png`
- `reports/figures/metrics_comparison.png`

## Key Concepts

### Preprocessing Pipeline
1. **clean_text**: lowercase, strip HTML, remove punctuation, collapse whitespace
2. **tokenize**: split into words using NLTK
3. **remove_stopwords**: drop "the", "is", "a", etc.
4. **lemmatize**: "running" → "run", "better" → "good"

### TF-IDF with N-grams
- **TF** (Term Frequency): how often a word appears in this document
- **IDF** (Inverse Document Frequency): how rare the word is across all documents
- **Bigrams**: capture phrases like "not good" as a single feature (preserves negation)

### Overfitting vs Underfitting
- **Overfit**: train score >> test score (gap > 0.10) → model memorised training data
- **Underfit**: both scores < 0.70 → model too simple, not learning enough
- **Good fit**: scores close and high → model generalises well

## Evaluation Metrics

| Metric | Formula | When to use |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Balanced datasets |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F1** | 2 × (P × R) / (P + R) | Imbalanced datasets |

All metrics are macro-averaged (each class weighted equally).

## Model Comparison

After training, the pipeline prints a comparison table:

```
===========================================================================
  MODEL COMPARISON — All models trained on identical TF-IDF features
===========================================================================
  Model                   Accuracy  Precision   Recall       F1    Diagnosis
  -----------------------------------------------------------------------
  naive_bayes               0.9500     0.9545   0.9500   0.9499     good_fit ◀ best
  logistic_regression       0.9500     0.9545   0.9500   0.9499     good_fit
  svm                       0.9500     0.9545   0.9500   0.9499     good_fit
  decision_tree             0.7500     0.7525   0.7500   0.7494      overfit
  random_forest             0.9000     0.9167   0.9000   0.8990     good_fit
  -----------------------------------------------------------------------
```

## Interview-Ready Notes

See `notes/learning_log.md` for:
- Concept definitions
- Code examples
- Interview questions with answers

Topics covered:
- TF-IDF and n-grams
- Preprocessing (stopwords, lemmatization)
- Model training and comparison
- Evaluation metrics
- Overfitting/underfitting
- Model persistence and inference

## Testing

### Run unit tests (pytest)

Always run via `python3 -m pytest` to ensure `src` imports resolve correctly:

```bash
python3 -m pytest tests/ -v
```

Run a specific test file:

```bash
python3 -m pytest tests/test_inference.py -v
```

### Run standalone inference demo

```bash
python3 test_inference.py
```

This tests all 5 models on sample positive, negative, and neutral texts and prints a results table.

## Dependencies

- Python 3.10+
- pandas, numpy
- scikit-learn
- nltk
- joblib
- matplotlib, seaborn
- scipy
- pytest (testing)
- hypothesis (property-based testing)

See `requirements.txt` for exact versions.

## License

MIT

## Author

Built as an interview-ready, production-style NLP classification pipeline demonstrating:
- Modular architecture (single responsibility per module)
- Classical NLP techniques (TF-IDF, preprocessing)
- Multiple classifier comparison
- Comprehensive evaluation and diagnostics
- Clean, explainable code
