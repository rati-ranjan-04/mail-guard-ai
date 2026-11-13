"""Capabilities & limits: what is implemented, what is external, what is not claimed."""

import streamlit as st

from mailguard import external, ui
from mailguard.security import esc

IMPLEMENTED = [
    ("Spam vs legitimate classification", "Multinomial Naive Bayes on TF-IDF (1-2 grams), probabilities calibrated with Platt scaling."),
    ("Explainable predictions", "Exact per-term contributions to the NB log-odds, plus a rule-by-rule list of findings."),
    ("URL extraction & analysis", "IP hosts, '@' tricks, punycode/non-ASCII hosts, shorteners, abused TLDs, brand look-alikes, "
                                  "href/text mismatch, risky file types, odd ports, embedded redirects. Offline string analysis only."),
    ("Sender indicators", "Display-name spoofing, brand mismatch, look-alike domains, free-mail 'official' senders, Reply-To / Return-Path mismatch."),
    ("SPF / DKIM / DMARC *result parsing*", "Reads the verdicts your mail server recorded in Authentication-Results / Received-SPF, with alignment checks."),
    ("Phishing language & attachment names", "Keyword/structure heuristics (credential requests, threats, urgency, OTP requests, double extensions, executables)."),
    ("Risk score 0-100 with severity", "Documented weighted sum with a visible breakdown: Low 0-24, Medium 25-49, High 50-74, Critical 75-100."),
    ("Batch CSV, history, PDF reports", "Local processing; history stores hashes and scores only; PDFs contain no email body."),
]
EXTERNAL = [
    ("Google Safe Browsing", "Needs GOOGLE_SAFE_BROWSING_API_KEY. Sends only the URL string. Opt-in per scan."),
    ("VirusTotal URL lookup", "Needs VIRUSTOTAL_API_KEY. Looks up an existing report; never submits new URLs for scanning. Opt-in per scan."),
]
UNSUPPORTED = [
    ("No trained phishing model", "The dataset has spam/ham labels only, so phishing indicators are heuristics and have not been benchmarked."),
    ("No URL fetching or sandboxing", "No redirect following or page rendering. A 'clean' URL result is not proof of safety."),
    ("No live DNS / cryptographic verification", "SPF/DKIM/DMARC values are only as trustworthy as the server that wrote the headers."),
    ("No attachment malware scanning", "Attachments are only described by name, type and size."),
    ("No reputation feeds", "No sender/IP reputation, domain-age (WHOIS) data or threat-intelligence subscriptions."),
    ("No mailbox integration", "No real-time protection; not a replacement for an enterprise email security gateway."),
    ("Limited evaluation data", "Measured on one public spam/ham corpus; may not hold on modern, targeted or non-English email."),
]


def _items(rows, cls):
    return "".join(f'<div class="mg-cap {cls}"><b>{esc(t)}</b><br><span class="mg-muted">{esc(d)}</span></div>' for t, d in rows)


def render():
    ui.hero("Capabilities & limits", "What this platform does and does not claim",
            "A clear separation between checks that run locally, optional external reputation lookups, and things that are not supported.")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### ✔ Implemented (offline)")
        st.markdown(_items(IMPLEMENTED, "ok"), unsafe_allow_html=True)
    with b:
        st.markdown("#### ☁ Optional external")
        st.markdown(_items(EXTERNAL, "opt"), unsafe_allow_html=True)
        prov = external.configured_providers()
        st.success("Configured: " + ", ".join(prov)) if prov else st.info("No API keys configured: running fully offline.")
        st.markdown("#### 🔑 Credentials")
        st.caption("API keys are read only from environment variables (or a git-ignored .env). They are never logged, shown or stored.")
    with c:
        st.markdown("#### ✖ Not supported / not claimed")
        st.markdown(_items(UNSUPPORTED, "no"), unsafe_allow_html=True)

    st.markdown("### Security design")
    st.markdown("""
- **Never opens links, never runs attachments.** URLs are strings; attachments are metadata only.
- **Input validation.** Text, CSV (5 MB, 2,000 rows) and .eml (2 MB) limits; binary content rejected; CSV exports neutralise spreadsheet formulas.
- **Output escaping.** Every dynamic value placed in HTML is escaped; URLs are displayed defanged.
- **Privacy-aware logging.** A logging filter masks e-mail addresses, URLs and key=value credentials; scans log only verdict and counts.
- **Model integrity.** The pipeline's SHA-256 is verified against `models/manifest.json` (`STRICT_MODEL_INTEGRITY=1` refuses on mismatch). Only load model files you trained yourself - joblib files are pickles.
- **Risk score weights are design choices**, not learned values; the breakdown is always shown so you can judge them.
""")


render()
