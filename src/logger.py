"""
=================================================================
 logger.py
=================================================================
A small, reusable logging setup for the whole project.

Instead of scattering print() statements everywhere, every script
imports get_logger() from here. This gives us, for free:
    - Timestamps on every message
    - Log level filtering (e.g. hide DEBUG messages in production)
    - Logs written BOTH to the console AND to a file (logs/app.log),
      so you can look back at what happened after the fact -- useful
      when training runs unattended or the Streamlit app is deployed.

Usage in any other file:

    from logger import get_logger
    logger = get_logger(__name__)

    logger.info("Training started")
    logger.warning("Missing values found, dropping rows")
    logger.error("Model file not found")
=================================================================
"""

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_LEVEL, LOG_FILE

# Make sure the logs/ folder exists before we try to write to it.
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# A consistent format for every log line across the whole project:
# 2026-08-16 10:30:00 | INFO | train | Training started
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "spam_classification") -> logging.Logger:
    """
    Create (or fetch, if already created) a logger with the given name
    that writes to both the console and a rotating log file.

    `RotatingFileHandler` automatically starts a new log file once the
    current one hits 1 MB, and keeps up to 3 old ones -- so log files
    never grow forever and eat up disk space.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called more
    # than once for the same name (e.g. across multiple imports).
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console handler: prints to the terminal, same as print() would ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler: also saves every log line to logs/app.log ---
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
