"""Rule-based phishing / social-engineering indicators and attachment risk.

IMPORTANT: the bundled dataset contains spam and legitimate mail only - it has no
phishing labels. These indicators are transparent keyword/structure heuristics,
not a trained phishing model, and have not been benchmarked on labelled phishing data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .common import Flag, cap, severity_for

_RULES = [
    ("phish.credentials", "Asks for credentials or sensitive data", 20,
     r"(verify|confirm|update|validate|re-?enter)\s+(your\s+)?(account|password|login|credentials|identity|payment|billing|card|bank)"
     r"|(enter|provide|send|share)\s+(your\s+)?(password|pin|otp|one[- ]time|cvv|ssn|social security|card number)"),
    ("phish.account_threat", "Threatens account suspension or lock-out", 15,
     r"(account|access|card|profile)\s+(will\s+be\s+|has\s+been\s+|is\s+)?(suspended|locked|limited|closed|disabled|terminated|deactivated)"
     r"|unusual\s+(sign[- ]?in|activity|login)|suspicious\s+(activity|sign[- ]?in|login)"),
    ("phish.urgency", "Creates artificial urgency", 8,
     r"\b(urgent(ly)?|immediately|act now|within\s+\d+\s+(hours?|days?)|expires?\s+(today|soon|in)|final\s+(notice|warning)|last\s+warning|asap)\b"),
    ("phish.click_prompt", "Pushes the reader to click or open something", 8,
     r"click\s+(here|the\s+link|below|on\s+the)|open\s+the\s+attach|follow\s+the\s+link|log\s*in\s+(here|now|below)"),
    ("phish.money_lure", "Financial lure (prize, refund, transfer, gift card, crypto)", 10,
     r"\b(wire\s+transfer|gift\s+cards?|bitcoin|crypto(currency)?|lottery|prize|inheritance|refund|unclaimed|tax\s+rebate|western\s+union|beneficiary)\b"),
    ("phish.generic_greeting", "Generic greeting instead of your name", 5,
     r"\bdear\s+(customer|user|member|client|account\s+holder|valued|sir|madam|friend|beneficiary)"),
    ("phish.otp_share", "Asks you to share a one-time code", 20,
     r"(share|send|tell|read\s+out|forward)\s+(me\s+|us\s+)?(the\s+|your\s+)?(otp|verification\s+code|one[- ]time\s+(password|code)|security\s+code)"),
    ("phish.impersonation", "Claims to be IT/security/bank/management", 6,
     r"\b(it\s+(help\s?desk|support|department)|security\s+team|bank\s+(security|fraud)|ceo|managing\s+director|payroll)\b"),
]
_COMPILED = [(c, l, p, re.compile(rx, re.I)) for c, l, p, rx in _RULES]

EXECUTABLE_EXT = {"exe", "scr", "js", "jse", "vbs", "vbe", "bat", "cmd", "jar", "msi", "lnk", "hta",
                  "ps1", "iso", "img", "apk", "com", "pif", "wsf", "dll"}
MACRO_OR_WEB_EXT = {"docm", "xlsm", "pptm", "html", "htm", "svg", "xlam", "one"}
ARCHIVE_EXT = {"zip", "rar", "7z", "gz", "tar", "cab", "ace"}
DOC_EXT = {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "jpg", "png"}


@dataclass
class PhishingAnalysis:
    score: int = 0
    severity: str = "Low"
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "severity": self.severity,
                "flags": [f.to_dict() for f in self.flags]}


def analyze_attachments(attachments: list) -> list:
    """Flag risky attachment names/types. Attachments are never opened or saved."""
    flags: list[Flag] = []
    for att in attachments or []:
        name = (att.get("name") or "").lower()
        parts = name.rsplit(".", 2)
        ext = parts[-1] if "." in name else ""
        label = "attachment"
        if ext in EXECUTABLE_EXT:
            flags.append(Flag("attach.executable", "Executable/script attachment", 25, name[-40:]))
        elif ext in MACRO_OR_WEB_EXT:
            flags.append(Flag("attach.macro_or_html", "Macro-enabled or HTML attachment", 12, name[-40:]))
        elif ext in ARCHIVE_EXT:
            flags.append(Flag("attach.archive", "Archive attachment (contents not inspected)", 8, name[-40:]))
        if len(parts) == 3 and parts[-2] in DOC_EXT and ext in EXECUTABLE_EXT | MACRO_OR_WEB_EXT:
            flags.append(Flag("attach.double_extension", "Double extension disguises file type", 15, name[-40:]))
        _ = label
    return flags


def analyze_phishing(text: str, subject: str = "", attachments: list | None = None) -> PhishingAnalysis:
    flags: list[Flag] = []
    blob = f"{subject}\n{text}"[:100_000]
    for code, label, pts, rx in _COMPILED:
        m = rx.search(blob)
        if m:
            snippet = " ".join(m.group(0).split())[:60]
            flags.append(Flag(code, label, pts, snippet))
    letters = [c for c in blob if c.isalpha()]
    if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        flags.append(Flag("phish.shouting", "Mostly capital letters", 4))
    if blob.count("!") >= 4:
        flags.append(Flag("phish.exclamations", "Excessive exclamation marks", 3))
    flags.extend(analyze_attachments(attachments or []))
    score = int(cap(sum(f.points for f in flags)))
    return PhishingAnalysis(score=score, severity=severity_for(score), flags=flags)
