import json
import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from mailguard import external
from mailguard.calibration import Calibrator, expected_calibration_error
from mailguard.explain import explain_prediction, raw_log_odds
from mailguard.history import ScanHistory
from mailguard.registry import verify_model, write_manifest
from mailguard.report_pdf import build_batch_report, build_email_report
from mailguard.security import (RedactingFilter, ValidationError, clean_input_text, defang_url, esc,
                                neutralize_csv_cell, read_csv_safely, redact_text, refang_url,
                                sanitize_dataframe_for_export, validate_upload)
from preprocessing import clean_text


def test_explanation_decomposes_log_odds_exactly(engine):
    text = clean_text("Subject: free prize winner click here to claim your discount meeting tomorrow budget")
    ex = explain_prediction(engine.pipeline, text, top_n=1000)
    assert ex.log_odds == pytest.approx(float(raw_log_odds(engine.pipeline, [text])[0]), abs=1e-9)
    assert all(v > 0 for _, v in ex.spam_terms) and all(v < 0 for _, v in ex.ham_terms)


def test_calibrator_roundtrip_and_monotone(tmp_path):
    rng = np.random.default_rng(0)
    lo = rng.normal(0, 10, 500)
    y = (rng.random(500) < 1 / (1 + np.exp(-lo / 3))).astype(int)
    cal = Calibrator.fit(lo, y)
    cal.save(tmp_path / "c.json")
    again = Calibrator.load(tmp_path / "c.json")
    p = again.predict(np.linspace(-30, 30, 50))
    assert np.all(np.diff(p) >= 0) and p.min() >= 0 and p.max() <= 1
    assert Calibrator.load(tmp_path / "missing.json") is None


def test_ece_perfect_and_bad():
    assert expected_calibration_error([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])[0] == 0
    assert expected_calibration_error([0, 0, 0, 0], [1.0] * 4)[0] == 1.0


def test_calibration_not_worse_in_report():
    c = json.load(open("reports/evaluation.json"))["calibration"]
    assert c["brier_calibrated"] <= c["brier_raw"] + 1e-3


def test_reported_metrics_match_confusion_matrix():
    """Stored metrics must be internally consistent (no fabricated numbers)."""
    ev = json.load(open("reports/evaluation.json"))
    for m in ev["models"].values():
        cm = m["confusion"]
        n = sum(cm.values())
        assert m["accuracy"] == pytest.approx((cm["tp"] + cm["tn"]) / n)
        if cm["tp"] + cm["fp"]:
            assert m["precision"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fp"]))
        assert m["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]))


def test_clean_input_text_limits():
    with pytest.raises(ValidationError):
        clean_input_text("   ")
    with pytest.raises(ValidationError):
        clean_input_text("a" * 11, max_chars=10)
    with pytest.raises(ValidationError):
        clean_input_text(123)
    assert clean_input_text("hi\x00there") == "hithere"


def test_validate_upload_rules():
    assert validate_upload("a.csv", b"a,b\n1,2", (".csv",), 100)
    for name, data in (("a.exe", b"x"), ("a.csv", b""), ("a.csv", b"x" * 200), ("a.csv", b"MZ\x00\x00")):
        with pytest.raises(ValidationError):
            validate_upload(name, data, (".csv",), 100)


def test_read_csv_safely_truncates_and_rejects():
    df, trunc = read_csv_safely(b"text\n" + b"\n".join(b"row%d" % i for i in range(30)), max_rows=10)
    assert len(df) == 10 and trunc
    with pytest.raises(ValidationError):
        read_csv_safely(b"")


def test_csv_formula_injection_neutralised():
    assert neutralize_csv_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert neutralize_csv_cell("safe") == "safe"
    out = sanitize_dataframe_for_export(pd.DataFrame({"a": ["=1+1", "ok"]}))
    assert out["a"].tolist() == ["'=1+1", "ok"]


def test_html_escape_and_defang_roundtrip():
    assert esc("<script>x</script>") == "&lt;script&gt;x&lt;/script&gt;"
    d = defang_url("https://evil.example/a")
    assert d == "hxxps://evil[.]example/a" and refang_url(d) == "https://evil.example/a"


def test_log_redaction():
    assert redact_text("mail bob@corp.com url http://evil.example/x key=SECRET123") == "mail <email> url <url> key=<redacted>"
    h = logging.StreamHandler()
    h.addFilter(RedactingFilter())
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "from %s", ("bob@corp.com",), None)
    assert h.filter(rec) and rec.getMessage() == "from <email>"


def test_scan_does_not_log_email_content(engine, phish, caplog):
    with caplog.at_level(logging.DEBUG):
        engine.analyze_email(phish)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "paypa1" not in text and "verify your account" not in text


def test_history_privacy_and_clear(tmp_path, engine, phish):
    h = ScanHistory(tmp_path / "h.db")
    h.add(engine.analyze_email(phish))
    h.add_url_scan({"url": "http://x.example/a", "severity": "Low", "score": 0, "domain": "x.example", "flags": []})
    dump = pd.read_sql_query("select * from scans", sqlite3.connect(tmp_path / "h.db")).to_json()
    assert "verify your account" not in dump and "/login" not in dump and "Urgent" not in dump
    assert len(h.list()) == 2 and h.count() == 2
    h.clear()
    assert h.count() == 0


def test_history_prunes(tmp_path, engine, ham):
    h = ScanHistory(tmp_path / "h.db", max_rows=3)
    r = engine.analyze_email(ham)
    for _ in range(6):
        h.add(r)
    assert h.count() == 3


def test_pdf_reports_valid(engine, phish):
    pdf = build_email_report(engine.analyze_email(phish))
    assert pdf[:5] == b"%PDF-" and len(pdf) > 2000
    summ = {"total": 2, "avg_risk": 50.0, "by_severity": {"High": 1, "Low": 1}, "by_verdict": {"Spam": 1},
            "model_version": "t", "truncated": False}
    row = {"row": 1, "risk": 90, "severity": "High", "verdict": "Spam", "indicators": "<b>x</b> é€"}
    assert build_batch_report(summ, [row])[:5] == b"%PDF-"


def test_pdf_handles_hostile_text(engine):
    res = engine.analyze_email('From: "<b>x</b>&" <a@b.example>\nSubject: <i>t</i>\nMIME-Version: 1.0\n\nbody http://x.example/<u>')
    assert build_email_report(res)[:5] == b"%PDF-"


def test_manifest_verify_detects_tamper(tmp_path):
    m, mf = tmp_path / "m.joblib", tmp_path / "manifest.json"
    m.write_bytes(b"abc")
    assert verify_model(m, mf)[0] == "unverified"
    write_manifest(m, mf, {"version": "v"})
    assert verify_model(m, mf)[0] == "verified"
    m.write_bytes(b"abd")
    assert verify_model(m, mf)[0] == "mismatch"
    assert verify_model(tmp_path / "none", mf)[0] == "missing"


def test_strict_integrity_refuses_tampered_model(tmp_path):
    import shutil
    import config
    from mailguard.engine import MailGuardEngine, ModelIntegrityError
    m, mf = tmp_path / "m.joblib", tmp_path / "mf.json"
    shutil.copy(config.MODEL_PATH, m)
    write_manifest(m, mf)
    m.write_bytes(m.read_bytes() + b"x")
    with pytest.raises(ModelIntegrityError):
        MailGuardEngine(m, tmp_path / "none.json", mf, strict=True)
    with pytest.raises(FileNotFoundError):
        MailGuardEngine(tmp_path / "nope.joblib", None, mf)


def test_external_disabled_without_keys(monkeypatch):
    monkeypatch.setattr(external.config, "VIRUSTOTAL_API_KEY", "")
    monkeypatch.setattr(external.config, "GOOGLE_SAFE_BROWSING_API_KEY", "")
    assert external.configured_providers() == [] and external.check_url("http://x.example") == []


def test_external_flag_and_error_handling(monkeypatch):
    monkeypatch.setattr(external.config, "GOOGLE_SAFE_BROWSING_API_KEY", "k")
    monkeypatch.setattr(external.config, "VIRUSTOTAL_API_KEY", "k2")

    def fake_http(req):
        if "safebrowsing" in req.full_url:
            return 200, {"matches": [{"threatType": "SOCIAL_ENGINEERING"}]}
        raise TimeoutError("boom")

    monkeypatch.setattr(external, "_http", fake_http)
    res = external.check_url("http://x.example")
    assert {r["provider"]: r["status"] for r in res} == {"Google Safe Browsing": "flagged", "VirusTotal": "error"}
    assert external.any_flagged({"u": res})
    assert "k2" not in json.dumps(res)


def test_external_never_called_by_default(engine, phish, monkeypatch):
    monkeypatch.setattr(external, "_http", lambda r: (_ for _ in ()).throw(AssertionError("network used")))
    assert engine.analyze_email(phish)["external"] is None
