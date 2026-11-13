"""Offline URL extraction and risk analysis.

URLs are only parsed as strings. They are never fetched, resolved or opened.
Scores are transparent heuristics (see ``Flag`` points), not a reputation feed.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import urlsplit, parse_qs

from .common import Flag, cap, severity_for
from .security import defang_url, refang_url
from .brands import BRAND_DOMAINS, brand_lookalike, registered_domain

URL_RE = re.compile(r"""(?i)\b((?:https?://|hxxps?://|www\.)[^\s<>"'`]+)""")
TRAILING = ".,;:!?)]}>'\""

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "t.ly", "lnkd.in", "s.id", "bl.ink",
}
SUSPICIOUS_TLDS = {
    "zip", "mov", "top", "xyz", "click", "work", "country", "gq", "tk", "ml", "cf", "ga",
    "rest", "support", "loan", "men", "review", "kim", "monster", "cfd", "sbs", "icu",
    "buzz", "cyou", "quest", "surf", "bid", "stream", "download", "racing", "win",
}
SENSITIVE_WORDS = ("login", "signin", "sign-in", "verify", "secure", "account", "update",
                   "password", "banking", "confirm", "wallet", "suspend", "unlock", "billing")
RISKY_EXT = (".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".jar", ".msi", ".apk", ".lnk",
             ".iso", ".hta", ".ps1", ".docm", ".xlsm")


@dataclass
class UrlFinding:
    url: str
    defanged: str
    host: str
    domain: str
    score: int
    severity: str
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"url": self.url, "defanged": self.defanged, "host": self.host,
                "domain": self.domain, "score": self.score, "severity": self.severity,
                "flags": [f.to_dict() for f in self.flags]}


def extract_urls(text: str, limit: int = 200) -> list[str]:
    """Find http(s)/www URLs in text (also hxxp-defanged ones). De-duplicated, order kept."""
    seen, out = set(), []
    for m in URL_RE.finditer(text or ""):
        url = refang_url(m.group(1)).rstrip(TRAILING)
        if url.lower().startswith("www."):
            url = "http://" + url
        if url and url not in seen:
            seen.add(url)
            out.append(url)
            if len(out) >= limit:
                break
    return out


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    # decimal (http://3232235777) or hex (0xC0A80001) encoded IPv4
    return bool(re.fullmatch(r"\d{8,10}", host) or re.fullmatch(r"0x[0-9a-f]{6,8}", host, re.I))


def _visible_domain(text: str):
    m = re.search(r"(?i)\b((?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,})\b", text or "")
    if not m:
        return None
    host = urlsplit(m.group(1) if "//" in m.group(1) else "//" + m.group(1)).hostname
    return registered_domain(host) if host else None


def analyze_url(url: str, anchor_text: str | None = None) -> UrlFinding:
    """Analyse one URL string offline and return explainable findings."""
    raw = refang_url(url)
    flags: list[Flag] = []

    def add(code, label, pts, detail=""):
        flags.append(Flag(code, label, pts, detail))

    lowered = raw.lower().strip()
    if lowered.startswith(("javascript:", "data:", "vbscript:")):
        add("url.script_scheme", "Script/data URL scheme", 50, lowered.split(":", 1)[0])
        return UrlFinding(raw[:200], defang_url(raw[:200]), "", "", 50, severity_for(50), flags)

    target = raw if "://" in raw else "http://" + raw
    try:
        parts = urlsplit(target)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        add("url.malformed", "Malformed URL", 25, "could not be parsed")
        return UrlFinding(raw[:200], defang_url(raw[:200]), "", "", 25, severity_for(25), flags)

    scheme = parts.scheme.lower()
    domain = registered_domain(host) if host and not _is_ip_host(host) else host
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    path_q = (parts.path or "") + ("?" + parts.query if parts.query else "")

    if not host:
        add("url.no_host", "URL has no host", 25)
    if _is_ip_host(host):
        add("url.ip_host", "Uses an IP address instead of a domain name", 30, host)
    if "@" in parts.netloc:
        add("url.userinfo", "Contains '@' (user-info trick hides the real host)", 25)
    if any(ord(c) > 127 for c in host):
        scripts = {unicodedata.name(c, "?").split()[0] for c in host if ord(c) > 127}
        add("url.non_ascii_host", "Non-ASCII (look-alike) characters in host", 25, ", ".join(sorted(scripts)))
    if "xn--" in host:
        add("url.punycode", "Punycode (internationalised) domain", 20, host)
    if domain in SHORTENERS:
        add("url.shortener", "URL shortener hides the final destination", 15, domain)
    if tld in SUSPICIOUS_TLDS:
        add("url.suspicious_tld", "Top-level domain frequently abused", 15, "." + tld)
    if scheme == "http":
        add("url.no_https", "Not using HTTPS", 10)
    if port and port not in (80, 443):
        add("url.odd_port", "Non-standard port", 10, str(port))
    if host and not _is_ip_host(host):
        extra_labels = len(host.split(".")) - len((domain or "").split("."))
        if extra_labels >= 3:
            add("url.many_subdomains", "Excessive subdomains", 10, host)
        if host.replace("xn--", "").count("-") >= 3:
            add("url.many_hyphens", "Many hyphens in host name", 8)
        digits = sum(c.isdigit() for c in host)
        if digits >= 6 and digits / max(1, len(host)) > 0.3:
            add("url.digit_heavy", "Digit-heavy host name", 8)
    if len(raw) > 200:
        add("url.very_long", "Very long URL", 10, f"{len(raw)} chars")
    elif len(raw) > 100:
        add("url.long", "Long URL", 5, f"{len(raw)} chars")
    if any(w in host for w in SENSITIVE_WORDS):
        add("url.sensitive_host", "Sensitive keyword in host name", 15)
    elif any(w in path_q.lower() for w in SENSITIVE_WORDS):
        add("url.sensitive_path", "Sensitive keyword in URL path", 8)
    if (parts.path or "").lower().endswith(RISKY_EXT):
        add("url.risky_file", "Links directly to an executable/macro file type", 20, parts.path[-30:])
    if any(re.match(r"(?i)https?://", v) for vals in parse_qs(parts.query).values() for v in vals):
        add("url.redirect_param", "Contains an embedded redirect URL", 10)

    if host and not _is_ip_host(host):
        label = (domain or "").split(".")[0]
        brand = brand_lookalike(label)
        official = BRAND_DOMAINS.get(brand[0], ()) if brand else ()
        if brand and domain not in official:
            add("url.brand_lookalike", "Look-alike of a well-known brand domain", 35, f"{label} ~ {brand[0]}")
        else:
            for name, official_domains in BRAND_DOMAINS.items():
                if len(name) >= 5 and name in host and domain not in official_domains:
                    add("url.brand_in_host", "Brand name used in an unrelated domain", 30, name)
                    break

    if anchor_text:
        shown = _visible_domain(anchor_text)
        if shown and domain and shown != domain:
            add("url.text_mismatch", "Link text shows a different domain than the real target", 30,
                f"{shown} vs {domain}")

    score = int(cap(sum(f.points for f in flags)))
    return UrlFinding(raw[:500], defang_url(raw[:500]), host, domain or "", score, severity_for(score), flags)


def analyze_urls(urls, anchors: dict | None = None, limit: int = 100) -> list[UrlFinding]:
    anchors = anchors or {}
    return [analyze_url(u, anchors.get(u)) for u in list(urls)[:limit]]
