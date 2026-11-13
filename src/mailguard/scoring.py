"""Transparent 0-100 threat-risk score.

The score is a *documented weighted sum*, not a learned model: every component
contributes points up to a fixed maximum, and the breakdown is shown to the user.
Weights are hand-set design choices (see README) and have not been tuned on
labelled phishing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import SEVERITY_ADVICE, cap, severity_for

# component -> maximum points it can add (sums to >100 on purpose; total is clamped)
MAX_POINTS = {"ml": 40, "phishing": 25, "url": 20, "sender": 10, "auth": 10, "external": 30}
SATURATION = 60  # a heuristic sub-score of 60+ earns the component's full points

COMPONENT_LABELS = {
    "ml": "Naive Bayes spam model (calibrated probability)",
    "phishing": "Phishing / social-engineering language & attachments",
    "url": "URL risk indicators",
    "sender": "Sender address & domain indicators",
    "auth": "SPF / DKIM / DMARC results (from headers)",
    "external": "External reputation check (optional)",
}


@dataclass
class Risk:
    score: int
    severity: str
    verdict: str
    breakdown: list = field(default_factory=list)
    advice: str = ""

    def to_dict(self) -> dict:
        return {"score": self.score, "severity": self.severity, "verdict": self.verdict,
                "breakdown": self.breakdown, "advice": self.advice}


def url_subscore(url_scores: list) -> int:
    """Worst URL dominates; each extra risky URL adds a little (max +15)."""
    if not url_scores:
        return 0
    worst = max(url_scores)
    extra = sum(1 for s in url_scores if s >= 25) - (1 if worst >= 25 else 0)
    return int(cap(worst + min(15, 5 * max(0, extra))))


def compute_risk(ml_prob: float, phishing_score: int, url_scores: list, sender_score: int,
                 auth_score: int, sender_assessed: bool, auth_assessed: bool,
                 external_flagged: bool | None = None, auth_failed: bool = False) -> Risk:
    url_sub = url_subscore(url_scores)
    parts = {
        "ml": (MAX_POINTS["ml"] * float(ml_prob), True, f"spam probability {ml_prob:.1%}"),
        "phishing": (MAX_POINTS["phishing"] * min(1, phishing_score / SATURATION), True,
                     f"heuristic sub-score {phishing_score}/100"),
        "url": (MAX_POINTS["url"] * min(1, url_sub / SATURATION), True,
                f"{len(url_scores)} URL(s), sub-score {url_sub}/100"),
        "sender": (MAX_POINTS["sender"] * min(1, sender_score / SATURATION), sender_assessed,
                   f"sub-score {sender_score}/100" if sender_assessed else "not assessed (no From header)"),
        "auth": (MAX_POINTS["auth"] * min(1, auth_score / SATURATION), auth_assessed,
                 f"sub-score {auth_score}/100" if auth_assessed else "not assessed (no authentication headers)"),
        "external": (MAX_POINTS["external"] if external_flagged else 0.0, external_flagged is not None,
                     "flagged by a provider" if external_flagged else
                     ("no provider flagged the URLs" if external_flagged is not None else "not run")),
    }
    breakdown, total = [], 0.0
    for key, (pts, assessed, note) in parts.items():
        pts = round(pts, 1) if assessed else 0.0
        total += pts
        breakdown.append({"component": key, "label": COMPONENT_LABELS[key], "points": pts,
                          "max_points": MAX_POINTS[key], "assessed": assessed, "note": note})
    score = int(round(cap(total)))
    severity = severity_for(score)

    strong_phish = phishing_score >= 25 or url_sub >= 40 or auth_failed or bool(external_flagged)
    if score >= 50 and strong_phish:
        verdict = "Likely phishing"
    elif ml_prob >= 0.5:
        verdict = "Spam"
    elif score >= 25:
        verdict = "Suspicious"
    else:
        verdict = "Legitimate (low risk)"
    return Risk(score, severity, verdict, breakdown, SEVERITY_ADVICE[severity])
