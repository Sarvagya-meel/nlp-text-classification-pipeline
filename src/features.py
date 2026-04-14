# src/features.py
# TF-IDF vectorization. Fit on training data only — never refit on test/inference.

import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    ngram_range: tuple = (1, 2),
    max_features: int = 10000,
    stop_words: str | None = "english",
) -> TfidfVectorizer:
    """Instantiate a TfidfVectorizer with the given hyperparameters."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=stop_words,
    )


def fit_transform_tfidf(
    vectorizer: TfidfVectorizer, X_train: pd.Series
) -> scipy.sparse.csr_matrix:
    """Fit vectorizer on X_train and return the transformed sparse matrix."""
    return vectorizer.fit_transform(X_train)


def transform_tfidf(
    vectorizer: TfidfVectorizer, X: pd.Series
) -> scipy.sparse.csr_matrix:
    """Transform X using an already-fitted vectorizer (no refitting)."""
    return vectorizer.transform(X)
