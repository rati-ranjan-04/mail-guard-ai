"""Mail Guard AI - email security and phishing detection platform (Streamlit entry point).

Run:  python -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mailguard import __version__, ui  # noqa: E402

st.set_page_config(page_title="Mail Guard AI", page_icon=":material/shield:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/email_scanner.py", title="Email scanner", icon=":material/mail:"),
    st.Page("views/url_scanner.py", title="URL scanner", icon=":material/link:"),
    st.Page("views/batch.py", title="Batch analysis", icon=":material/table_view:"),
    st.Page("views/history.py", title="Scan history", icon=":material/history:"),
    st.Page("views/analytics.py", title="Analytics & model", icon=":material/monitoring:"),
    st.Page("views/capabilities.py", title="Capabilities & limits", icon=":material/verified_user:"),
]
nav = st.navigation(pages)
with st.sidebar:
    ui.brand()
    st.caption(f"v{__version__} · Multinomial Naive Bayes core")
    st.caption("Theme: menu (top right) → Settings → Light / Dark / System")
    st.caption("Decision-support tool, not a replacement for an enterprise email gateway.")

nav.run()
