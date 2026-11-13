"""SPF / DKIM / DMARC *result parsing* from message headers.

This module reads authentication verdicts that the receiving mail server already
recorded (``Authentication-Results`` / ``Received-SPF``). It does NOT perform DNS
lookups or cryptographic DKIM verification itself. Only the top-most
``Authentication-Results`` header is trusted: lower ones may be forged by the sender.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .brands import registered_domain
from .common import Flag, cap, severity_for

KNOWN = {"pass", "fail", "softfail", "neutral", "none", "temperror", "permerror", "policy", "hardfail"}


@dataclass
class AuthResult:
    assessed: bool = False
    spf: str = "missing"
    dkim: str = "missing"
    dmarc: str = "missing"
    dkim_domains: list = field(default_factory=list)
    spf_domain: str = ""
    score: int = 0
    severity: str = "Low"
    flags: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"assessed": self.assessed, "spf": self.spf, "dkim": self.dkim, "dmarc": self.dmarc,
                "dkim_domains": self.dkim_domains, "spf_domain": self.spf_domain,
                "score": self.score, "severity": self.severity, "note": self.note,
                "flags": [f.to_dict() for f in self.flags]}


def _method(text: str, name: str) -> str | None:
    m = re.search(rf"(?i)(?<![\w.-]){name}\s*=\s*([a-z]+)", text)
    if m and m.group(1).lower() in KNOWN:
        return m.group(1).lower()
    return None


def analyze_authentication(auth_results: list, received_spf: list, dkim_signatures: list,
                           from_domain: str = "") -> AuthResult:
    res = AuthResult()
    flags: list[Flag] = []
    header = auth_results[0] if auth_results else ""

    spf = _method(header, "spf") if header else None
    dkim = _method(header, "dkim") if header else None
    dmarc = _method(header, "dmarc") if header else None
    if spf is None and received_spf:
        m = re.match(r"(?i)\s*([a-z]+)", received_spf[0])
        if m and m.group(1).lower() in KNOWN:
            spf = m.group(1).lower()
    sig_domains = []
    for sig in dkim_signatures:
        m = re.search(r"(?i)\bd=([A-Za-z0-9.-]+)", sig)
        if m:
            sig_domains.append(m.group(1).lower())
    res.dkim_domains = sig_domains

    if not header and not received_spf:
        res.note = ("No Authentication-Results/Received-SPF header found. SPF/DKIM/DMARC "
                    "were not assessed (this is not a failure).")
        return res

    res.assessed = True
    res.spf, res.dkim, res.dmarc = spf or "missing", dkim or "missing", dmarc or "missing"
    m = re.search(r"(?i)smtp\.mailfrom\s*=\s*(?:[^@\s;]*@)?([A-Za-z0-9.-]+)", header)
    res.spf_domain = m.group(1).lower() if m else ""
    m = re.search(r"(?i)header\.d\s*=\s*([A-Za-z0-9.-]+)", header)
    if m:
        res.dkim_domains = sorted(set(res.dkim_domains + [m.group(1).lower()]))

    def add(code, label, pts, detail=""):
        flags.append(Flag(code, label, pts, detail))

    spf_pts = {"fail": 25, "hardfail": 25, "softfail": 15, "permerror": 15, "temperror": 5,
               "none": 8, "missing": 5, "neutral": 5}
    if res.spf in spf_pts:
        add(f"auth.spf_{res.spf}", f"SPF result: {res.spf}", spf_pts[res.spf])
    dkim_pts = {"fail": 25, "permerror": 15, "temperror": 5, "none": 8, "missing": 8, "neutral": 5}
    if res.dkim in dkim_pts:
        add(f"auth.dkim_{res.dkim}", f"DKIM result: {res.dkim}", dkim_pts[res.dkim])
    dmarc_pts = {"fail": 30, "permerror": 15, "temperror": 5, "none": 8, "missing": 8}
    if res.dmarc in dmarc_pts:
        add(f"auth.dmarc_{res.dmarc}", f"DMARC result: {res.dmarc}", dmarc_pts[res.dmarc])

    fd = registered_domain(from_domain) if from_domain else ""
    if fd:
        if res.dkim == "pass" and res.dkim_domains and all(registered_domain(d) != fd for d in res.dkim_domains):
            add("auth.dkim_misaligned", "DKIM passed for a domain different from the From domain", 10,
                ",".join(res.dkim_domains))
        if res.spf == "pass" and res.spf_domain and registered_domain(res.spf_domain) != fd:
            add("auth.spf_misaligned", "SPF passed for a domain different from the From domain", 8, res.spf_domain)

    res.flags = flags
    res.score = int(cap(sum(f.points for f in flags)))
    res.severity = severity_for(res.score)
    res.note = ("Parsed from headers recorded by the receiving server; no DNS or cryptographic "
                "verification was performed here.")
    return res
