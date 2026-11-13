"""OPTIONAL external URL-reputation lookups (network + API key required).

These are NOT part of the offline detection engine. They only run when the user
explicitly opts in AND a key is configured via environment variables. Only the URL
string is sent to the provider; the URL itself is never visited from this app.
Keys are never logged or displayed. Not live-verified in this repository's test run
(no keys/network in CI); behaviour is covered by mocked tests.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import config
from logger import get_logger

logger = get_logger(__name__)

TIMEOUT = 8
MAX_URLS = 10
_SB_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
_VT_ENDPOINT = "https://www.virustotal.com/api/v3/urls/"


def configured_providers() -> list[str]:
    out = []
    if config.GOOGLE_SAFE_BROWSING_API_KEY:
        out.append("Google Safe Browsing")
    if config.VIRUSTOTAL_API_KEY:
        out.append("VirusTotal")
    return out


def _http(req: urllib.request.Request) -> tuple[int, dict]:
    if not req.full_url.startswith("https://"):
        raise ValueError("Only HTTPS provider endpoints are allowed.")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https hosts
            return resp.status, json.loads(resp.read(1_000_000).decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def _safe_browsing(url: str) -> dict:
    body = {
        "client": {"clientId": "mail-guard-ai", "clientVersion": "2.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"], "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    req = urllib.request.Request(
        f"{_SB_ENDPOINT}?key={config.GOOGLE_SAFE_BROWSING_API_KEY}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    status, data = _http(req)
    if status != 200:
        return {"provider": "Google Safe Browsing", "status": "error", "detail": f"HTTP {status}"}
    matches = data.get("matches") or []
    if matches:
        kinds = sorted({m.get("threatType", "?") for m in matches})
        return {"provider": "Google Safe Browsing", "status": "flagged", "detail": ", ".join(kinds)}
    return {"provider": "Google Safe Browsing", "status": "clean", "detail": "no match in Safe Browsing lists"}


def _virustotal(url: str) -> dict:
    ident = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    req = urllib.request.Request(_VT_ENDPOINT + ident, headers={"x-apikey": config.VIRUSTOTAL_API_KEY})
    status, data = _http(req)
    if status == 404:
        return {"provider": "VirusTotal", "status": "unknown", "detail": "URL not previously analysed (not submitted)"}
    if status != 200:
        return {"provider": "VirusTotal", "status": "error", "detail": f"HTTP {status}"}
    stats = ((data.get("data") or {}).get("attributes") or {}).get("last_analysis_stats") or {}
    bad = int(stats.get("malicious", 0)) + int(stats.get("suspicious", 0))
    if bad > 0:
        return {"provider": "VirusTotal", "status": "flagged", "detail": f"{bad} engine(s) flagged it"}
    return {"provider": "VirusTotal", "status": "clean", "detail": "no engine flagged it"}


_PROVIDERS = {"Google Safe Browsing": _safe_browsing, "VirusTotal": _virustotal}


def check_url(url: str) -> list[dict]:
    """Query every configured provider for one URL. Never raises; errors become results."""
    results = []
    for name in configured_providers():
        try:
            results.append(_PROVIDERS[name](url))
        except Exception as exc:  # network/JSON errors; log only the type (no URL, no key)
            logger.warning("External reputation check failed: %s", type(exc).__name__)
            results.append({"provider": name, "status": "error", "detail": type(exc).__name__})
    return results


def check_urls(urls: list[str]) -> dict[str, list[dict]]:
    return {u: check_url(u) for u in urls[:MAX_URLS]}


def any_flagged(results: dict[str, list[dict]]) -> bool:
    return any(r["status"] == "flagged" for rs in results.values() for r in rs)
