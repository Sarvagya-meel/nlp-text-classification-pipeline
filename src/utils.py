# src/utils.py
# Shared helpers used across modules.

import os
import nltk


def ensure_dir(path: str) -> None:
    """Create directory at path if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already present."""
    for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
