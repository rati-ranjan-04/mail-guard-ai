"""
=================================================================
 config.py
=================================================================
Single source of truth for all file paths and settings used across
the project. Reads values from the .env file (via python-dotenv) so
that train.py, predict.py, and app.py never hard-code paths directly
-- they all import from here instead.

If a value isn't found in .env, a sensible default is used, so the
project still works even if someone forgets to create a .env file.
=================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root (this file lives at project_root/src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load variables from the .env file at the project root, if present.
# This does nothing (silently) if no .env file exists.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve(path_str: str) -> Path:
    """Turn a relative path from .env into an absolute path from project root."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --- Data paths ---
RAW_DATA_PATH = _resolve(os.getenv("RAW_DATA_PATH", "data/raw/emails.csv"))
PROCESSED_DATA_PATH = _resolve(
    os.getenv("PROCESSED_DATA_PATH", "data/processed/emails_cleaned.csv")
)

# --- Model path ---
MODEL_PATH = _resolve(os.getenv("MODEL_PATH", "models/spam_classifier_pipeline.joblib"))

# --- Logging settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = _resolve(os.getenv("LOG_FILE", "logs/app.log"))
