"""Privacy-preserving scan history (SQLite).

Stores ONLY: timestamp, kind, SHA-256 of the content, length, verdict, scores,
severity, sender domain, counts and generic indicator labels. It never stores the
email subject, body, addresses' local parts or URL paths.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, kind TEXT NOT NULL, content_sha256 TEXT, length INTEGER,
    verdict TEXT, risk INTEGER, severity TEXT, spam_probability REAL,
    sender_domain TEXT, n_urls INTEGER, indicators TEXT, model_version TEXT
)"""
COLUMNS = ["id", "ts", "kind", "content_sha256", "length", "verdict", "risk", "severity",
           "spam_probability", "sender_domain", "n_urls", "indicators", "model_version"]


class ScanHistory:
    def __init__(self, path: Path | None = None, max_rows: int | None = None):
        self.path = Path(path or config.HISTORY_DB)
        self.max_rows = max_rows or config.HISTORY_MAX_ROWS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as con, con:
            con.execute(_SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, result: dict, kind: str = "email") -> int:
        meta, risk = result["meta"], result["risk"]
        indicators = [r["label"] for r in result.get("reasons", [])[:8]]
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, meta.get("content_sha256"),
               int(meta.get("length", 0)), risk["verdict"], int(risk["score"]), risk["severity"],
               float(result["ml"]["spam_probability"]), (result.get("sender") or {}).get("domain", ""),
               len(result.get("urls", [])), json.dumps(indicators), meta.get("model_version", ""))
        with closing(self._conn()) as con, con:
            cur = con.execute("INSERT INTO scans (ts,kind,content_sha256,length,verdict,risk,severity,"
                              "spam_probability,sender_domain,n_urls,indicators,model_version) "
                              "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
            con.execute("DELETE FROM scans WHERE id NOT IN (SELECT id FROM scans ORDER BY id DESC LIMIT ?)",
                        (self.max_rows,))
            return int(cur.lastrowid)

    def add_url_scan(self, finding: dict) -> int:
        fake = {"meta": {"content_sha256": "", "length": len(finding["url"]), "model_version": ""},
                "risk": {"verdict": "URL: " + finding["severity"], "score": finding["score"],
                         "severity": finding["severity"]},
                "ml": {"spam_probability": 0.0}, "sender": {"domain": finding["domain"]}, "urls": [1],
                "reasons": [{"label": f["label"]} for f in finding["flags"]]}
        return self.add(fake, kind="url")

    def list(self, limit: int = 500) -> pd.DataFrame:
        with closing(self._conn()) as con:
            df = pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC LIMIT ?", con, params=(int(limit),))
        if not df.empty:
            df["indicators"] = df["indicators"].map(lambda s: "; ".join(json.loads(s or "[]")))
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def count(self) -> int:
        with closing(self._conn()) as con:
            return int(con.execute("SELECT COUNT(*) FROM scans").fetchone()[0])

    def clear(self) -> None:
        with closing(self._conn()) as con, con:
            con.execute("DELETE FROM scans")
            con.execute("DELETE FROM sqlite_sequence WHERE name='scans'")
