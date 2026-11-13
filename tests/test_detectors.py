import pytest

from mailguard.common import severity_for
from mailguard.headers import analyze_authentication
from mailguard.parsing import html_to_text, parse_email
from mailguard.phishing import analyze_attachments, analyze_phishing
from mailguard.scoring import compute_risk, url_subscore
from mailguard.sender import analyze_sender
from mailguard.urls import analyze_url, extract_urls


def codes(f):
    return {x.code for x in f.flags}


def test_url_ip_and_exe():
    f = analyze_url("http://192.168.1.5:8080/a.exe")
    assert {"url.ip_host", "url.odd_port", "url.risky_file", "url.no_https"} <= codes(f)
    assert f.severity in ("High", "Critical")


def test_url_lookalike_and_official():
    assert "url.brand_lookalike" in codes(analyze_url("http://paypa1-secure.xyz/login"))
    assert analyze_url("https://www.paypal.com/").score == 0
    assert analyze_url("https://mail.google.com/x").score == 0


def test_url_no_false_lookalike_for_common_words():
    assert "url.brand_lookalike" not in codes(analyze_url("https://ample.com"))


def test_url_userinfo_punycode_shortener():
    assert "url.userinfo" in codes(analyze_url("https://paypal.com@evil.example/x"))
    assert "url.punycode" in codes(analyze_url("http://xn--pple-43d.com"))
    assert "url.shortener" in codes(analyze_url("https://bit.ly/abc"))


def test_url_text_mismatch():
    f = analyze_url("https://evil.example/x", anchor_text="https://www.paypal.com/login")
    assert "url.text_mismatch" in codes(f)


def test_url_script_scheme_and_malformed():
    assert analyze_url("javascript:alert(1)").score >= 50
    assert analyze_url("http://[::1").score > 0


def test_extract_urls_defanged_and_dedupe():
    urls = extract_urls("see www.foo.com/a, hxxp://bad[.]com/x. and www.foo.com/a")
    assert urls == ["http://www.foo.com/a", "http://bad.com/x"]


def test_url_defanged_never_clickable():
    f = analyze_url("http://evil.example/a")
    assert f.defanged.startswith("hxxp://") and "[.]" in f.defanged


def test_sender_spoof_signals():
    s = analyze_sender('"PayPal Support" <x@paypa1-help.example>', reply_to="a@other.test")
    c = {f.code for f in s.flags}
    assert {"sender.brand_mismatch", "sender.reply_to_mismatch"} <= c


def test_sender_legit_and_missing():
    assert analyze_sender("Alice <alice@campus.example>").score == 0
    assert analyze_sender("").assessed is False
    assert analyze_sender("garbage").flags[0].code == "sender.invalid"


def test_auth_parsing_and_not_assessed():
    a = analyze_authentication(["mx; spf=fail smtp.mailfrom=x.example; dkim=none; dmarc=fail header.from=x.example"], [], [], "x.example")
    assert (a.spf, a.dkim, a.dmarc) == ("fail", "none", "fail") and a.score >= 50
    none = analyze_authentication([], [], [], "x.example")
    assert none.assessed is False and none.score == 0


def test_auth_only_top_header_trusted_and_alignment():
    a = analyze_authentication(["mx; spf=pass smtp.mailfrom=bounce.other.test; dkim=pass header.d=other.test; dmarc=pass",
                                "forged; spf=fail"], [], [], "brand.example")
    assert a.spf == "pass"
    assert {f.code for f in a.flags} >= {"auth.dkim_misaligned", "auth.spf_misaligned"}


def test_phishing_rules():
    r = analyze_phishing("Dear customer, verify your password now. Your account will be suspended. Click here")
    assert r.score >= 40
    assert analyze_phishing("Lunch at noon, see you there").score == 0


def test_attachments_double_extension():
    f = analyze_attachments([{"name": "invoice.pdf.exe"}, {"name": "notes.txt"}])
    assert {x.code for x in f} == {"attach.executable", "attach.double_extension"}


def test_parse_plain_and_dataset_style():
    p = parse_email("Subject: hello\nbody text")
    assert (p.subject, p.body_text, p.has_headers) == ("hello", "body text", False)
    assert parse_email("Subject: all on one line").body_text == "Subject: all on one line"


def test_parse_headers_html_and_attachment():
    raw = ("From: A <a@x.example>\nTo: b@y.example\nSubject: Hi\nMIME-Version: 1.0\n"
           'Content-Type: multipart/mixed; boundary="B"\n\n--B\nContent-Type: text/html\n\n'
           '<p>Go <a href="http://evil.example/x">www.paypal.com</a></p><script>bad()</script>\n'
           '--B\nContent-Type: application/octet-stream\nContent-Disposition: attachment; filename="a.exe"\n\nMZ\n--B--\n')
    p = parse_email(raw)
    assert p.has_headers and p.has_html and "bad()" not in p.body_text
    assert p.html_links == [("http://evil.example/x", "www.paypal.com")]
    assert p.attachments[0]["name"] == "a.exe"


def test_html_to_text_drops_script():
    t, _ = html_to_text("<style>x{}</style><p>Hi</p><script>evil()</script>")
    assert t == "Hi"


@pytest.mark.parametrize("score,sev", [(0, "Low"), (24, "Low"), (25, "Medium"), (49, "Medium"), (50, "High"), (74, "High"), (75, "Critical"), (100, "Critical")])
def test_severity_boundaries(score, sev):
    assert severity_for(score) == sev


def test_risk_unassessed_adds_zero_and_clamps():
    r = compute_risk(0.0, 0, [], 0, 0, False, False)
    assert r.score == 0 and r.verdict.startswith("Legitimate")
    top = compute_risk(1.0, 100, [100, 100], 100, 100, True, True, True, True)
    assert top.score == 100 and top.verdict == "Likely phishing"
    assert all(b["points"] == 0 for b in r.breakdown)


def test_url_subscore_extra_urls():
    assert url_subscore([]) == 0
    assert url_subscore([60, 30, 30]) == 70
