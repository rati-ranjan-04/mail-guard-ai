"""MailGuardEngine: orchestrates parsing, ML, explanation, heuristics and scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

import config
from preprocessing import clean_text
from logger import get_logger

from . import external as ext
from .calibration import Calibrator
from .explain import explain_prediction, raw_log_odds
from .headers import analyze_authentication
from .parsing import ParsedEmail, parse_email
from .phishing import analyze_phishing
from .registry import read_manifest, verify_model
from .scoring import compute_risk
from .security import clean_input_text
from .sender import analyze_sender
from .urls import analyze_urls, extract_urls

logger = get_logger(__name__)


class ModelIntegrityError(RuntimeError):
    pass


class MailGuardEngine:
    def __init__(self, model_path: Path | None = None, calibration_path: Path | None = None,
                 manifest_path: Path | None = None, strict: bool | None = None):
        model_path = Path(model_path or config.MODEL_PATH)
        manifest_path = Path(manifest_path or config.MANIFEST_PATH)
        strict = config.STRICT_MODEL_INTEGRITY if strict is None else strict
        status, message = verify_model(model_path, manifest_path)
        if status == "missing":
            raise FileNotFoundError(f"Model not found at {model_path.name}. Run `python src/train.py` first.")
        if status == "mismatch" and strict:
            raise ModelIntegrityError(message)
        if status in ("mismatch", "unverified"):
            logger.warning("Model integrity: %s", message)
        self.integrity_status, self.integrity_message = status, message
        self.pipeline = joblib.load(model_path)  # pickle: only load files you trained yourself
        self.calibrator = Calibrator.load(Path(calibration_path or config.CALIBRATION_PATH))
        manifest = read_manifest(manifest_path) or {}
        if self.calibrator and self.calibrator.meta.get("for_model_sha"):
            from .registry import sha256_file
            if self.calibrator.meta["for_model_sha"] != sha256_file(model_path):
                logger.warning("Calibration file was fitted for a different model; ignoring it.")
                self.calibrator = None
        self.manifest = manifest
        self.model_version = manifest.get("version", "unversioned")
        self.model_sha = manifest.get("sha256", "")[:12]
        logger.info("Engine ready (model %s, integrity=%s, calibrated=%s)",
                    self.model_version, status, self.calibrator is not None)

    # ------------------------------------------------------------------ ML
    @property
    def probability_source(self) -> str:
        return "calibrated" if self.calibrator else "raw Naive Bayes score (uncalibrated)"

    def ml_scores(self, cleaned_texts: list[str]) -> dict:
        """Vectorised scoring. Returns spam probability (calibrated if available)."""
        lo = raw_log_odds(self.pipeline, cleaned_texts)
        raw_p = self.pipeline.predict_proba(cleaned_texts)[:, 1]
        prob = self.calibrator.predict(lo) if self.calibrator else raw_p
        return {"raw_probability": raw_p, "probability": np.asarray(prob), "log_odds": lo}

    # ------------------------------------------------------------ assemble
    def _assemble(self, parsed: ParsedEmail, ml_prob: float, raw_prob: float, log_odds: float,
                  explanation: dict | None, ext_results: dict | None) -> dict:
        urls_text = extract_urls(parsed.body_text + "\n" + parsed.subject)
        anchors = {}
        for href, text in parsed.html_links:
            if href.lower().startswith(("http://", "https://", "javascript:", "data:", "//")):
                href = "http:" + href if href.startswith("//") else href
                if href not in urls_text:
                    urls_text.append(href)
                if text:
                    anchors[href] = text
        findings = analyze_urls(urls_text, anchors)

        sender = analyze_sender(parsed.from_header, parsed.reply_to, parsed.return_path)
        auth = analyze_authentication(parsed.auth_results, parsed.received_spf, parsed.dkim_signatures, sender.domain)
        phishing = analyze_phishing(parsed.body_text, parsed.subject, parsed.attachments)

        ext_flagged = ext.any_flagged(ext_results) if ext_results is not None else None
        auth_failed = auth.assessed and (auth.dmarc == "fail" or auth.spf in ("fail", "hardfail"))
        risk = compute_risk(ml_prob, phishing.score, [f.score for f in findings], sender.score, auth.score,
                            sender.assessed, auth.assessed, ext_flagged, auth_failed)

        reasons = []
        for source, flags in (("Phishing language", phishing.flags), ("Sender", sender.flags),
                              ("Authentication", auth.flags)):
            reasons += [{"source": source, **f.to_dict()} for f in flags]
        for f in findings:
            reasons += [{"source": "URL", **fl.to_dict(), "detail": f"{f.defanged} - {fl.detail}".strip(" -")}
                        for fl in f.flags]
        reasons.sort(key=lambda r: -r["points"])

        return {
            "risk": risk.to_dict(),
            "ml": {"label": "Spam" if ml_prob >= 0.5 else "Ham (not spam)",
                   "spam_probability": float(ml_prob), "raw_probability": float(raw_prob),
                   "log_odds": float(log_odds), "probability_source": self.probability_source,
                   "explanation": explanation},
            "urls": [f.to_dict() for f in findings],
            "sender": sender.to_dict(),
            "auth": auth.to_dict(),
            "phishing": phishing.to_dict(),
            "attachments": parsed.attachments,
            "reasons": reasons,
            "external": ext_results,
            "meta": {"model_version": self.model_version, "model_sha": self.model_sha,
                     "integrity": self.integrity_status, "has_headers": parsed.has_headers,
                     "has_html": parsed.has_html, "truncated": parsed.truncated,
                     "length": len(parsed.body_text) + len(parsed.subject),
                     "content_sha256": hashlib.sha256(parsed.ml_text.encode("utf-8", "replace")).hexdigest()},
        }

    # ------------------------------------------------------------- public
    def analyze_email(self, raw, use_external: bool = False) -> dict:
        raw = clean_input_text(raw) if isinstance(raw, str) else raw
        parsed = parse_email(raw)
        cleaned = clean_text(parsed.ml_text)
        s = self.ml_scores([cleaned])
        expl = explain_prediction(self.pipeline, cleaned).to_dict()
        ext_results = None
        if use_external and ext.configured_providers():
            ext_results = ext.check_urls(extract_urls(parsed.body_text))
        result = self._assemble(parsed, float(s["probability"][0]), float(s["raw_probability"][0]),
                                float(s["log_odds"][0]), expl, ext_results)
        logger.info("Scan complete: verdict=%s score=%s severity=%s urls=%d", result["risk"]["verdict"],
                    result["risk"]["score"], result["risk"]["severity"], len(result["urls"]))
        return result

    def analyze_batch(self, rows: list[dict], progress=None) -> list[dict]:
        """rows: [{'text':..., 'sender': optional, 'subject': optional}]. ML is vectorised."""
        parsed_rows = []
        for r in rows:
            p = parse_email(r.get("text", ""))
            if r.get("subject") and not p.subject:
                p.subject = r["subject"]
            if r.get("sender") and not p.from_header:
                p.from_header = r["sender"]
            parsed_rows.append(p)
        cleaned = [clean_text(p.ml_text) for p in parsed_rows]
        s = self.ml_scores(cleaned) if cleaned else {"probability": [], "raw_probability": [], "log_odds": []}
        out = []
        for i, p in enumerate(parsed_rows):
            out.append(self._assemble(p, float(s["probability"][i]), float(s["raw_probability"][i]),
                                      float(s["log_odds"][i]), None, None))
            if progress:
                progress((i + 1) / len(parsed_rows))
        logger.info("Batch scan complete: %d rows", len(out))
        return out

    def analyze_urls_only(self, urls: list[str], use_external: bool = False) -> dict:
        findings = analyze_urls(urls)
        ext_results = ext.check_urls([f.url for f in findings]) if use_external and ext.configured_providers() else None
        return {"urls": [f.to_dict() for f in findings], "external": ext_results}
