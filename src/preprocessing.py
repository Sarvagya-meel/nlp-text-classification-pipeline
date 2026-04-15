# src/preprocessing.py
# Text cleaning and normalization. No I/O — pure transformations only.
#
# WHY PREPROCESSING MATTERS
# ─────────────────────────────────────────────────────────────────────────────
# Raw text is noisy. Without preprocessing, the TF-IDF vocabulary fills up with
# junk tokens ("the", "is", "!!!", "<br>") that carry no signal, and meaningful
# words appear as separate tokens ("run", "running", "ran") even though they
# represent the same concept. Preprocessing fixes both problems.
#
# PIPELINE ORDER (matters — each step feeds the next):
#   1. clean_text        → normalize casing, strip noise
#   2. tokenize          → split into individual words
#   3. remove_stopwords  → drop high-frequency, low-signal words
#   4. lemmatize/stem    → collapse morphological variants
#   5. rejoin            → return clean string for TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove punctuation, collapse whitespace.

    WHY:
    - Lowercasing ensures "Good" and "good" are treated as the same token.
      Without it, the vocabulary doubles unnecessarily.
    - HTML removal strips tags like <br>, <b> that appear in web-scraped data
      and add no semantic value.
    - Punctuation removal prevents "great!" and "great" from being counted
      as different features in TF-IDF.
    - Whitespace normalization ensures consistent tokenization downstream.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # strip HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()    # collapse multiple spaces
    return text


def remove_missing(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop rows where col is null or empty; raise ValueError if result is empty."""
    df = df.copy()
    df = df[df[col].notna()]
    df = df[df[col].str.strip() != ""]
    if df.empty:
        raise ValueError(f"No valid rows remain after removing missing values in '{col}'.")
    return df.reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    """Split text into individual word tokens using NLTK word_tokenize.

    WHY:
    - Simple str.split() misses edge cases (contractions, hyphenated words).
    - NLTK's tokenizer handles "don't" → ["do", "n't"] and similar cases
      correctly, giving cleaner input to stopword removal and lemmatization.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list[str], lang: str = "english") -> list[str]:
    """Remove common stopwords from a token list.

    WHY:
    - Stopwords ("the", "is", "and", "a", "in") appear in almost every
      document. They dominate TF-IDF vocabulary but carry zero class signal.
    - Removing them shrinks the feature space, reduces noise, and forces
      the model to focus on words that actually distinguish classes.
    - Example: "this is a great product" → ["great", "product"]
      The model now sees the meaningful words, not the filler.

    TRADE-OFF:
    - Occasionally stopwords carry sentiment (e.g. "not bad" → removing "not"
      flips the meaning). For most classification tasks the gain outweighs
      this risk, but be aware of negation in sentiment tasks.
    """
    stop_words = set(stopwords.words(lang))
    return [t for t in tokens if t not in stop_words]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to reduce words to their root form.

    WHY:
    - Stemming aggressively chops word endings: "running" → "run",
      "happiness" → "happi". This reduces vocabulary size significantly.
    - Faster than lemmatization but produces non-real-word stems ("happi").
    - Best used when speed matters more than linguistic accuracy.

    TRADE-OFF vs LEMMATIZATION:
    - Stemming is faster but cruder. "studies" → "studi" (not a real word).
    - Lemmatization is slower but produces real words: "studies" → "study".
    - For interview-level NLP, lemmatization is generally preferred.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply WordNet lemmatization to reduce words to their dictionary base form.

    WHY:
    - Lemmatization maps inflected forms to a single canonical token:
        "running" → "run", "better" → "good", "studies" → "study"
    - This means "run", "runs", "running", "ran" all become "run" in the
      TF-IDF vocabulary — one feature instead of four.
    - Reduces vocabulary size → fewer features → less overfitting.
    - Improves recall: the model recognises "running" and "ran" as the
      same concept it learned from "run" in training data.

    PERFORMANCE IMPACT:
    - Smaller, cleaner vocabulary → TF-IDF matrix is denser with signal.
    - Particularly helpful on small datasets where each token needs to
      carry maximum information.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_series(
    series: pd.Series,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> pd.Series:
    """Apply the full preprocessing pipeline to a pandas text Series.

    Pipeline applied per document:
      clean_text → tokenize → remove_stopwords → lemmatize (or stem) → rejoin

    Args:
        series:           Raw text Series (nulls must be removed beforehand).
        use_stemming:     Apply PorterStemmer (fast, crude).
        use_lemmatization: Apply WordNetLemmatizer (slower, linguistically correct).

    Note: use_stemming and use_lemmatization are mutually exclusive.
    Default is lemmatization — better quality for classification tasks.
    """
    if use_stemming and use_lemmatization:
        raise ValueError("use_stemming and use_lemmatization cannot both be True.")

    def _process(text: str) -> str:
        # Step 1: normalize casing, remove HTML and punctuation
        text = clean_text(text)

        # Step 2: split into tokens for word-level operations
        tokens = tokenize(text)

        # Step 3: drop stopwords — removes noise, shrinks feature space
        tokens = remove_stopwords(tokens)

        # Step 4: morphological normalization — collapse word variants
        if use_stemming:
            tokens = stem_tokens(tokens)       # fast but crude
        elif use_lemmatization:
            tokens = lemmatize_tokens(tokens)  # slower but linguistically correct

        # Step 5: rejoin into a string for TF-IDF input
        return " ".join(tokens)

    return series.apply(_process)
