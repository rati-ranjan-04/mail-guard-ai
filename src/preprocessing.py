"""
preprocessing.py
-----------------
Shared text-cleaning utilities used consistently during both
model training (train.py) and inference (predict.py / app.py).

Keeping this logic in ONE place avoids train/serve skew and
guarantees the exact same preprocessing is applied every time.
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Normalize a single raw email text string.

    Steps:
    1. Lowercase everything (case shouldn't change meaning for spam detection).
    2. Strip the leading 'Subject:' token that prefixes every record in this dataset.
    3. Collapse repeated whitespace into single spaces.
    4. Strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"^subject:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the raw emails.csv, remove duplicates/empties, and clean the text column.
    Returns a cleaned DataFrame with columns ['text', 'spam'].
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df
