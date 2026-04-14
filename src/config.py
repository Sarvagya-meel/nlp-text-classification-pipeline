# src/config.py
# Central configuration — all constants live here, never hardcoded elsewhere.

import os

# --- Paths ---
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
METRICS_PATH = "reports/metrics/"
FIGURES_PATH = "reports/figures/"

# --- Column names ---
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# --- Train/test split ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- TF-IDF ---
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 10000
TFIDF_STOP_WORDS = "english"

TFIDF_PARAMS = {
    "ngram_range": TFIDF_NGRAM_RANGE,
    "max_features": TFIDF_MAX_FEATURES,
    "stop_words": TFIDF_STOP_WORDS,
}
