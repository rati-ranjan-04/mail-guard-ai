from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["overview", "email_scanner", "url_scanner", "batch", "history", "analytics", "capabilities"]


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "HISTORY_DB", tmp_path / "h.db")
    from mailguard import ui
    ui.get_history.clear()
    yield
    ui.get_history.clear()


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()


def test_full_app_navigation_has_no_exceptions():
    at = _app()
    assert not at.exception
    for page in PAGES:
        at.switch_page(f"views/{page}.py").run()
        assert not at.exception, page


def test_email_scan_flow_and_state():
    from mailguard.samples import SAMPLES
    at = _app()
    at.switch_page("views/email_scanner.py").run()
    at.session_state["email_text"] = SAMPLES["Phishing-style (synthetic)"]
    next(b for b in at.button if b.key == "scan_paste").click().run()
    assert not at.exception and not at.error
    assert at.session_state["email_result"]["risk"]["severity"] in ("High", "Critical")


def test_empty_input_shows_error_not_crash():
    at = _app()
    at.switch_page("views/email_scanner.py").run()
    next(b for b in at.button if b.key == "scan_paste").click().run()
    assert not at.exception and any("empty" in e.value.lower() for e in at.error)


def test_url_scanner_flow():
    at = _app()
    at.switch_page("views/url_scanner.py").run()
    at.text_area[0].set_value("http://paypa1-secure.xyz/login\nhxxp://1[.]2[.]3[.]4/a.exe").run()
    next(b for b in at.button if b.label == "Analyse URLs").click().run()
    assert not at.exception
    assert len(at.session_state["url_result"]["urls"]) == 2


def test_app_works_without_reportlab(monkeypatch):
    """A missing optional PDF dependency must not crash the app."""
    import sys
    from mailguard.samples import SAMPLES
    for m in [m for m in sys.modules if m.startswith(("reportlab", "mailguard.report_pdf"))]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setitem(sys.modules, "reportlab", None)
    at = _app()
    at.switch_page("views/email_scanner.py").run()
    at.session_state["email_text"] = SAMPLES["Legitimate-style (synthetic)"]
    next(b for b in at.button if b.key == "scan_paste").click().run()
    assert not at.exception
    assert any("reportlab" in w.value for w in at.warning)
