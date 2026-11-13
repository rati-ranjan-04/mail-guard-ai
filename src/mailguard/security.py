"""Input validation, safe upload handling, output sanitising, log redaction.

Nothing here opens URLs, executes attachments or evals content.
"""

from __future__ import annotations

import html
import io
import logging
import re

import pandas as pd

MAX_TEXT_CHARS = 200_000
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_EML_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 2_000
MAX_CSV_COLUMNS = 50
MAX_CELL_CHARS = 50_000

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ValidationError(ValueError):
    """Raised for rejected user input. Messages are safe to show in the UI."""


def clean_input_text(text, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Validate pasted text: type, emptiness, length. Strips control chars."""
    if not isinstance(text, str):
        raise ValidationError("Input must be text.")
    text = _CONTROL_RE.sub("", text)
    if not text.strip():
        raise ValidationError("Input is empty.")
    if len(text) > max_chars:
        raise ValidationError(
            f"Input is too large ({len(text):,} characters; limit {max_chars:,})."
        )
    return text


def validate_upload(filename: str, data: bytes, allowed_ext: tuple, max_bytes: int) -> bytes:
    """Check extension, size, emptiness and binary content of an upload."""
    name = (filename or "").lower()
    if not name.endswith(tuple(allowed_ext)):
        raise ValidationError(f"Unsupported file type. Allowed: {', '.join(allowed_ext)}.")
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(
            f"File is too large ({len(data) / 1e6:.1f} MB; limit {max_bytes / 1e6:.0f} MB)."
        )
    if b"\x00" in data[:8192]:
        raise ValidationError("File looks binary, not text. It was not processed.")
    return data


def decode_bytes(data: bytes) -> str:
    """Decode uploaded bytes as UTF-8 (with BOM tolerance), falling back to latin-1."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_csv_safely(data: bytes, max_rows: int = MAX_BATCH_ROWS):
    """Parse CSV bytes into an all-text DataFrame. Returns (df, was_truncated).

    Only parses data: no formulas are evaluated, no paths are followed.
    """
    text = decode_bytes(data)
    try:
        df = pd.read_csv(
            io.StringIO(text), sep=None, engine="python", dtype=str,
            nrows=max_rows + 1, on_bad_lines="skip",
        )
    except Exception as exc:  # pandas raises several types
        raise ValidationError("Could not parse the file as CSV.") from exc
    if df.empty or len(df.columns) == 0:
        raise ValidationError("The CSV has no data rows.")
    if len(df.columns) > MAX_CSV_COLUMNS:
        raise ValidationError(f"Too many columns ({len(df.columns)}; limit {MAX_CSV_COLUMNS}).")
    truncated = len(df) > max_rows
    df = df.head(max_rows).copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].str.slice(0, MAX_CELL_CHARS)
    return df, truncated


def neutralize_csv_cell(value):
    """Prevent spreadsheet formula injection in exported CSV cells."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def sanitize_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].map(neutralize_csv_cell)
    return out


def esc(value) -> str:
    """HTML-escape any value before it is placed inside unsafe_allow_html markup."""
    return html.escape(str(value), quote=True)


def defang_url(url: str) -> str:
    """Make a URL non-clickable and non-auto-linkable: hxxp://example[.]com/..."""
    out = re.sub(r"^http", "hxxp", url, flags=re.I)
    out = re.sub(r"^ftp", "fxp", out, flags=re.I)
    # defang the host's dots only (keep readability of the path)
    m = re.match(r"^([a-z]+://)?([^/?#]*)(.*)$", out, flags=re.I | re.S)
    if m:
        scheme, host, rest = m.groups()
        out = f"{scheme or ''}{host.replace('.', '[.]')}{rest}"
    return out


def refang_url(url: str) -> str:
    """Reverse common defanging (hxxp, [.]) so analysts can paste defanged IOCs."""
    out = url.strip()
    out = re.sub(r"^hxxp", "http", out, flags=re.I)
    out = re.sub(r"^fxp", "ftp", out, flags=re.I)
    return out.replace("[.]", ".").replace("(.)", ".").replace("[:]", ":")


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_KEY_RE = re.compile(r"(?i)(key|token|secret|apikey|api_key)=([^&\s]+)")


def redact_text(text: str) -> str:
    """Mask e-mail addresses, URLs and key=value credentials in log text."""
    text = _KEY_RE.sub(r"\1=<redacted>", text)
    text = _URL_RE.sub("<url>", text)
    return _EMAIL_RE.sub("<email>", text)


class RedactingFilter(logging.Filter):
    """Logging filter: defence in depth so addresses/URLs/keys never reach the log."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_text(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True
