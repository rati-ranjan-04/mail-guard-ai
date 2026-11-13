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


# --- Mail Guard AI platform settings (added in v2) ---
CALIBRATION_PATH = _resolve(os.getenv("CALIBRATION_PATH", "models/calibration.json"))
MANIFEST_PATH = _resolve(os.getenv("MANIFEST_PATH", "models/manifest.json"))
VERSIONS_DIR = _resolve(os.getenv("VERSIONS_DIR", "models/versions"))
REPORTS_DIR = _resolve(os.getenv("REPORTS_DIR", "reports"))
HISTORY_DB = _resolve(os.getenv("HISTORY_DB", "data/scan_history.db"))
HISTORY_MAX_ROWS = int(os.getenv("HISTORY_MAX_ROWS", "10000"))
# Set to 1 to refuse to load a model whose SHA-256 differs from models/manifest.json
STRICT_MODEL_INTEGRITY = os.getenv("STRICT_MODEL_INTEGRITY", "0") == "1"

# Optional external reputation APIs. Read from the environment ONLY; never hard-code,
# print or log these. Leave empty to keep the app fully offline.
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
GOOGLE_SAFE_BROWSING_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")
