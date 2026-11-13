"""Shared Streamlit UI helpers: theme tokens, cards, badges, cached resources.

All dynamic text placed in unsafe HTML goes through ``esc`` (html.escape).
"""

from __future__ import annotations

import streamlit as st

from .security import esc

DARK = {
    "bg": "#0a1226", "card": "#121c38", "card2": "#17244a", "border": "#25325a", "text": "#e8edf8",
    "muted": "#a3b3d4", "accent": "#4c8dff", "track": "#25325a",
    "Low": "#34d399", "Medium": "#fbbf24", "High": "#fb923c", "Critical": "#f87171",
    "spam": "#d95926", "ham": "#3987e5",
}
LIGHT = {
    "bg": "#f3f6fc", "card": "#ffffff", "card2": "#eef2fa", "border": "#d3dcee", "text": "#0b1b3a",
    "muted": "#4b5b7c", "accent": "#1f5fd1", "track": "#dbe3f3",
    "Low": "#15803d", "Medium": "#a16207", "High": "#c2410c", "Critical": "#b91c1c",
    "spam": "#eb6834", "ham": "#2a78d6",
}
SEV_ICON = {"Low": "✔", "Medium": "▲", "High": "◆", "Critical": "✖"}  # shape + label: never colour alone


def theme_type() -> str:
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else "dark"
    except Exception:
        return "dark"


def tokens() -> dict:
    return LIGHT if theme_type() == "light" else DARK


def inject_css() -> None:
    c = tokens()
    st.markdown(f"""
<style>
:root {{
  --mg-card:{c['card']}; --mg-card2:{c['card2']}; --mg-border:{c['border']}; --mg-text:{c['text']};
  --mg-muted:{c['muted']}; --mg-accent:{c['accent']}; --mg-track:{c['track']};
  --mg-low:{c['Low']}; --mg-medium:{c['Medium']}; --mg-high:{c['High']}; --mg-critical:{c['Critical']};
}}
.block-container {{ padding-top: 3.6rem; max-width: 1240px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.mg-brand {{ display:flex; align-items:center; gap:.6rem; font-weight:700; font-size:1.15rem; }}
.mg-brand small {{ display:block; font-weight:500; font-size:.72rem; color:var(--mg-muted); }}
.mg-hero {{ border:1px solid var(--mg-border); border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1.1rem;
  background: linear-gradient(135deg, var(--mg-card) 0%, var(--mg-card2) 100%); }}
.mg-hero h1 {{ margin:0 0 .35rem; font-size:1.7rem; }}
.mg-hero p {{ margin:0; color:var(--mg-muted); max-width: 70ch; }}
.mg-kicker {{ text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; color:var(--mg-accent); margin-bottom:.4rem; }}
.mg-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.8rem; margin:.4rem 0 1rem; }}
.mg-card {{ background:var(--mg-card); border:1px solid var(--mg-border); border-radius:14px; padding:1rem 1.1rem; }}
.mg-card .lbl {{ color:var(--mg-muted); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }}
.mg-card .val {{ font-size:1.65rem; font-weight:750; line-height:1.25; margin-top:.15rem; }}
.mg-card .sub {{ color:var(--mg-muted); font-size:.8rem; margin-top:.1rem; }}
.mg-badge {{ display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .65rem; border-radius:999px;
  font-weight:700; font-size:.82rem; border:1.5px solid currentColor; }}
.mg-sev-Low {{ color:var(--mg-low); }} .mg-sev-Medium {{ color:var(--mg-medium); }}
.mg-sev-High {{ color:var(--mg-high); }} .mg-sev-Critical {{ color:var(--mg-critical); }}
.mg-result {{ display:grid; grid-template-columns:170px 1fr; gap:1.2rem; align-items:center; }}
@media (max-width: 640px) {{ .mg-result {{ grid-template-columns:1fr; justify-items:center; text-align:center; }} }}
.mg-verdict {{ font-size:1.35rem; font-weight:750; margin:.3rem 0; }}
.mg-muted {{ color:var(--mg-muted); }}
.mg-cap {{ border-left:4px solid var(--mg-border); padding:.15rem 0 .15rem .8rem; margin:.35rem 0; }}
.mg-cap.ok {{ border-color:var(--mg-low); }} .mg-cap.opt {{ border-color:var(--mg-medium); }} .mg-cap.no {{ border-color:var(--mg-critical); }}
.mg-pill {{ display:inline-block; padding:.1rem .5rem; border-radius:6px; background:var(--mg-card2);
  border:1px solid var(--mg-border); font-size:.78rem; margin:.1rem .15rem .1rem 0; }}
.mg-safe-note {{ border:1px dashed var(--mg-border); border-radius:10px; padding:.55rem .8rem; color:var(--mg-muted); font-size:.85rem; }}
[data-testid="stSidebarNav"] a {{ border-radius:8px; }}
</style>""", unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        '<div class="mg-brand"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M12 2l8 3v6c0 5-3.4 9.3-8 11-4.6-1.7-8-6-8-11V5l8-3z" fill="#4c8dff" fill-opacity=".18" '
        'stroke="#4c8dff" stroke-width="1.8" stroke-linejoin="round"/><path d="M8.5 12.3l2.4 2.4 4.6-5" '
        'stroke="#4c8dff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        '<div>Mail Guard AI<small>Email security platform</small></div></div>', unsafe_allow_html=True)


def hero(kicker: str, title: str, text: str) -> None:
    st.markdown(f'<div class="mg-hero"><div class="mg-kicker">{esc(kicker)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def kpi_grid(items: list[tuple]) -> None:
    """items: (label, value, sub). Responsive CSS grid of cards."""
    cards = "".join(f'<div class="mg-card"><div class="lbl">{esc(l)}</div><div class="val">{esc(v)}</div>'
                    f'<div class="sub">{esc(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="mg-grid">{cards}</div>', unsafe_allow_html=True)


def severity_badge(sev: str) -> str:
    return f'<span class="mg-badge mg-sev-{esc(sev)}">{SEV_ICON.get(sev, "")} {esc(sev)}</span>'


def score_ring(score: int, severity: str) -> str:
    circ = 2 * 3.14159 * 52
    dash = circ * max(0, min(100, score)) / 100
    return (f'<svg viewBox="0 0 120 120" width="160" height="160" role="img" aria-label="Risk score {int(score)} of 100, '
            f'{esc(severity)}"><circle cx="60" cy="60" r="52" fill="none" stroke="var(--mg-track)" stroke-width="11"/>'
            f'<circle class="mg-sev-{esc(severity)}" cx="60" cy="60" r="52" fill="none" stroke="currentColor" stroke-width="11" '
            f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" transform="rotate(-90 60 60)"/>'
            f'<text x="60" y="64" text-anchor="middle" font-size="30" font-weight="800" fill="var(--mg-text)">{int(score)}</text>'
            f'<text x="60" y="82" text-anchor="middle" font-size="9" fill="var(--mg-muted)">RISK / 100</text></svg>')


def result_header(result: dict) -> None:
    r = result["risk"]
    st.markdown(
        f'<div class="mg-card"><div class="mg-result"><div>{score_ring(r["score"], r["severity"])}</div><div>'
        f'{severity_badge(r["severity"])}<div class="mg-verdict">{esc(r["verdict"])}</div>'
        f'<div class="mg-muted">{esc(r["advice"])}</div></div></div></div>', unsafe_allow_html=True)


def not_visited_note() -> None:
    st.markdown('<div class="mg-safe-note">🔒 Analysis runs locally. Links are only parsed as text, never opened; '
                'attachments are never executed or saved. URLs are shown defanged (hxxp, [.]).</div>',
                unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine():
    from .engine import MailGuardEngine
    return MailGuardEngine()


@st.cache_resource(show_spinner=False)
def get_history():
    from .history import ScanHistory
    return ScanHistory()


def load_evaluation():
    import json
    import config
    path = config.REPORTS_DIR / "evaluation.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def require_engine():
    """Return the engine or show a friendly error and stop the page."""
    try:
        return get_engine()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
    except Exception as exc:
        from logger import get_logger
        get_logger(__name__).error("Engine failed to load: %s", type(exc).__name__)
        st.error("The model could not be loaded. If you changed scikit-learn versions, reinstall "
                 "requirements.txt or retrain with `python src/train.py`.")
        if type(exc).__name__ == "ModelIntegrityError":
            st.warning("Integrity check failed: the model file does not match models/manifest.json.")
    st.stop()
