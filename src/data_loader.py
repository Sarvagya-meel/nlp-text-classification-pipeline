# src/data_loader.py
# Responsible for loading raw CSV data and validating required columns.

import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return it as a DataFrame."""
    return pd.read_csv(filepath)


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError listing any required columns missing from df."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
