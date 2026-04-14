# src/preprocessing.py
# Text cleaning and normalization. No I/O — pure transformations only.

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
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
    """Tokenize text into a list of word tokens using NLTK."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str], lang: str = "english") -> list[str]:
    """Remove NLTK stopwords from a token list."""
    stop_words = set(stopwords.words(lang))
    return [t for t in tokens if t not in stop_words]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply PorterStemmer to each token."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply WordNetLemmatizer to each token."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_series(
    series: pd.Series,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> pd.Series:
    """Apply full preprocessing pipeline to a text Series."""
    if use_stemming and use_lemmatization:
        raise ValueError("use_stemming and use_lemmatization cannot both be True.")

    def _process(text: str) -> str:
        text = clean_text(text)
        tokens = tokenize(text)
        tokens = remove_stopwords(tokens)
        if use_stemming:
            tokens = stem_tokens(tokens)
        elif use_lemmatization:
            tokens = lemmatize_tokens(tokens)
        return " ".join(tokens)

    return series.apply(_process)
