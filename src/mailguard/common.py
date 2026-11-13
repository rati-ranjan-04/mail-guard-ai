"""Shared small types: indicator flags and severity levels."""

from __future__ import annotations

from dataclasses import dataclass, asdict

# (name, lower bound inclusive, colour token used by the UI)
SEVERITY_LEVELS = [
    ("Low", 0),
    ("Medium", 25),
    ("High", 50),
    ("Critical", 75),
]

SEVERITY_ADVICE = {
    "Low": "No strong risk indicators were found. Normal caution still applies: "
           "do not open unexpected attachments or links.",
    "Medium": "Some risk indicators were found. Verify the sender through a "
              "separate channel before clicking links or replying.",
    "High": "Multiple risk indicators were found. Do not click links, open "
            "attachments or reply. Report it to your IT/security team.",
    "Critical": "Strong risk indicators were found. Do not interact with this "
                "message. Report it and delete it; reset credentials if you already responded.",
}


def severity_for(score: float) -> str:
    """Map a 0-100 score to Low / Medium / High / Critical."""
    score = max(0.0, min(100.0, float(score)))
    name = SEVERITY_LEVELS[0][0]
    for level, lower in SEVERITY_LEVELS:
        if score >= lower:
            name = level
    return name


@dataclass
class Flag:
    """One explainable risk indicator."""

    code: str       # stable machine-readable id, safe to store in history
    label: str      # generic human-readable label (contains no email content)
    points: int     # contribution to the sub-score
    detail: str = ""  # may contain content-specific text; never stored/logged

    def to_dict(self) -> dict:
        return asdict(self)


def cap(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
