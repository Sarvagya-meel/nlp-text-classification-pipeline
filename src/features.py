# src/features.py
# TF-IDF vectorization with n-gram support.
# Fit on training data only — never refit on test or inference data.
#
# WHY TF-IDF?
# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF (Term Frequency–Inverse Document Frequency) converts raw text into
# a numerical matrix that ML models can consume.
#
# TF  (Term Frequency)         = how often a word appears in THIS document
# IDF (Inverse Doc Frequency)  = how rare the word is ACROSS all documents
# TF-IDF score                 = TF × IDF
#
# Words that appear often in one document but rarely across all documents
# get high scores — these are the discriminative, class-specific words.
# Common words like "the" appear everywhere → low IDF → low TF-IDF → ignored.
#
# WHY N-GRAMS?
# ─────────────────────────────────────────────────────────────────────────────
# Unigrams (n=1) treat each word independently: ["not", "good"]
# Bigrams  (n=2) capture word pairs:            ["not good"]
#
# "not good" as a bigram carries the opposite sentiment to "good" alone.
# Without bigrams, "not good" and "very good" look similar to the model
# because both contain "good". Bigrams preserve this context.
#
# ngram_range=(1, 2) includes BOTH unigrams and bigrams — best of both worlds:
# - Unigrams give broad vocabulary coverage
# - Bigrams capture local context and negation patterns
#
# TRADE-OFF:
# - Bigrams increase vocabulary size significantly (can be 5-10x larger).
# - max_features caps this to keep the matrix manageable.
# - Trigrams (n=3) rarely help for short texts and slow training considerably.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    ngram_range: tuple = (1, 2),
    max_features: int = 10000,
    stop_words: str | None = "english",
) -> TfidfVectorizer:
    """Instantiate a TfidfVectorizer with the given hyperparameters.

    Args:
        ngram_range:  (min_n, max_n) — (1,1) unigrams only, (1,2) adds bigrams.
                      Bigrams capture phrases like "not good", "highly recommend".
        max_features: Vocabulary size cap. Limits memory and prevents overfitting
                      on rare n-grams. 10,000 is a solid default for most tasks.
        stop_words:   Built-in stopword list. Set to None if stopwords were
                      already removed in preprocessing (avoids double filtering).
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=stop_words,
    )


def fit_transform_tfidf(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series,
) -> scipy.sparse.csr_matrix:
    """Fit vectorizer on X_train and return the transformed sparse matrix.

    WHY fit only on training data:
    - Fitting on test data would leak information about the test distribution
      into the vocabulary (data leakage), inflating evaluation scores.
    - The vectorizer learns which tokens exist and their IDF weights from
      training data only — test data is transformed using those learned weights.
    """
    return vectorizer.fit_transform(X_train)


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    X: pd.Series,
) -> scipy.sparse.csr_matrix:
    """Transform X using an already-fitted vectorizer without refitting.

    WHY never refit on test/inference data:
    - Refitting would change the vocabulary and IDF weights, making the
      transformation inconsistent with what the model was trained on.
    - Unseen words at inference time are simply ignored (out-of-vocabulary),
      which is the correct behaviour — not an error.
    """
    return vectorizer.transform(X)
