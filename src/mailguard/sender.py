"""Sender address / domain risk indicators (offline heuristics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parseaddr
import re

from .brands import (BRAND_DOMAINS, DISPOSABLE, FREE_MAIL, OFFICIAL_WORDS,
                     brand_lookalike, registered_domain)
from .common import Flag, cap, severity_for
from .urls import SUSPICIOUS_TLDS


@dataclass
class SenderAnalysis:
    display_name: str = ""
    address: str = ""
    domain: str = ""
    reply_to_domain: str = ""
    score: int = 0
    severity: str = "Low"
    flags: list = field(default_factory=list)
    assessed: bool = False

    def to_dict(self) -> dict:
        return {"display_name": self.display_name, "address": self.address, "domain": self.domain,
                "reply_to_domain": self.reply_to_domain, "score": self.score,
                "severity": self.severity, "assessed": self.assessed,
                "flags": [f.to_dict() for f in self.flags]}


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[1].lower().strip(" >.") if "@" in address else ""


def analyze_sender(from_header: str, reply_to: str = "", return_path: str = "") -> SenderAnalysis:
    """Evaluate From / Reply-To / Return-Path. Returns assessed=False when From is missing."""
    result = SenderAnalysis()
    if not (from_header or "").strip():
        return result
    result.assessed = True
    name, addr = parseaddr(from_header)
    addr = addr.strip().lower()
    domain = _domain_of(addr)
    result.display_name, result.address, result.domain = name.strip(), addr, domain
    flags: list[Flag] = []

    def add(code, label, pts, detail=""):
        flags.append(Flag(code, label, pts, detail))

    if not domain or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", addr):
        add("sender.invalid", "Missing or malformed sender address", 20)
        result.flags, result.score = flags, 20
        result.severity = severity_for(20)
        return result

    reg = registered_domain(domain)
    local = addr.split("@", 1)[0]
    lname = name.lower()

    # Display name embeds a different address/domain (classic spoof)
    m = re.search(r"[\w.+-]+@([\w.-]+\.[a-z]{2,})", lname)
    if m and registered_domain(m.group(1)) != reg:
        add("sender.display_spoof", "Display name shows a different e-mail address", 25, m.group(1))

    # Brand in display name but sender domain is not that brand's
    for brand, official in BRAND_DOMAINS.items():
        if brand in lname.replace(" ", "") and reg not in official:
            add("sender.brand_mismatch", "Display name claims a brand the domain does not belong to", 30, brand)
            break

    # Look-alike domain
    label = reg.split(".")[0]
    near = brand_lookalike(label)
    if near and reg not in BRAND_DOMAINS.get(near[0], ()):
        add("sender.lookalike_domain", "Sender domain imitates a well-known brand", 35, f"{label} ~ {near[0]}")
    else:
        for brand, official in BRAND_DOMAINS.items():
            if len(brand) >= 5 and brand in domain and reg not in official:
                add("sender.brand_in_domain", "Brand name embedded in an unrelated sender domain", 30, brand)
                break

    if reg in FREE_MAIL and (any(w in lname for w in OFFICIAL_WORDS) or any(w in local for w in OFFICIAL_WORDS)):
        add("sender.free_mail_official", "Free-mail address presented as an official/support sender", 20)
    if reg in DISPOSABLE:
        add("sender.disposable", "Disposable e-mail provider", 15)
    if any(ord(c) > 127 for c in domain) or "xn--" in domain:
        add("sender.idn", "Internationalised (possible look-alike) sender domain", 20)
    if domain.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
        add("sender.suspicious_tld", "Sender domain uses a frequently abused TLD", 15)
    if sum(c.isdigit() for c in domain) >= 5 or sum(c.isdigit() for c in local) >= 7:
        add("sender.digit_heavy", "Unusually many digits in sender address", 8)
    if re.search(r"(secure|login|verify|account|support|update)", reg) and "-" in reg:
        add("sender.keyword_domain", "Hyphenated 'security-style' sender domain", 12)

    rt = parseaddr(reply_to)[1].lower() if reply_to else ""
    if rt and _domain_of(rt):
        result.reply_to_domain = _domain_of(rt)
        if registered_domain(result.reply_to_domain) != reg:
            add("sender.reply_to_mismatch", "Reply-To domain differs from From domain", 20, result.reply_to_domain)
    rp = parseaddr(return_path)[1].lower() if return_path else ""
    if rp and _domain_of(rp) and registered_domain(_domain_of(rp)) != reg:
        add("sender.return_path_mismatch", "Return-Path domain differs from From domain", 8, _domain_of(rp))

    result.flags = flags
    result.score = int(cap(sum(f.points for f in flags)))
    result.severity = severity_for(result.score)
    return result
