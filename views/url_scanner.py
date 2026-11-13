"""URL scanner page (offline heuristics; optional external reputation)."""

import pandas as pd
import streamlit as st

from logger import get_logger
from mailguard import external, ui
from mailguard.security import ValidationError, clean_input_text, esc
from mailguard.urls import extract_urls

logger = get_logger(__name__)


def render():
    ui.hero("URL scanner", "Check suspicious links safely",
            "Paste one or more URLs (defanged hxxp/[.] forms are accepted). They are analysed as text only: "
            "never opened, resolved or followed.")
    ui.not_visited_note()
    st.write("")
    text = st.text_area("URLs (one per line, or any text containing links)", height=150, max_chars=20000,
                        placeholder="http://paypa1-security-check.example/login\nhxxps://bit[.]ly/abc123")
    providers = external.configured_providers()
    c1, c2 = st.columns([1, 1])
    with c1:
        use_ext = st.checkbox("Also run external reputation check", value=False, disabled=not providers,
                              help="Sends only the URL strings to the configured provider.")
        if not providers:
            st.caption("No API keys configured: offline heuristics only.")
    with c2:
        save = st.checkbox("Save to history (domain + score only)", value=True)

    if st.button("Analyse URLs", type="primary", icon=":material/link:"):
        engine = ui.require_engine()
        try:
            text = clean_input_text(text, max_chars=20000)
            urls = extract_urls(text) or extract_urls(" ".join("http://" + l.strip() for l in text.splitlines() if l.strip() and "://" not in l))
            if not urls:
                st.warning("No URLs were found in the input.")
            else:
                with st.spinner("Analysing URLs..."):
                    out = engine.analyze_urls_only(urls, use_external=use_ext)
                if save:
                    for f in out["urls"]:
                        ui.get_history().add_url_scan(f)
                st.session_state["url_result"] = out
                st.toast(f"Analysed {len(out['urls'])} URL(s)", icon="✅")
        except ValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.error("URL scan failed: %s", type(exc).__name__)
            st.error("The URL analysis failed unexpectedly.")

    out = st.session_state.get("url_result")
    if not out:
        st.info("Enter URLs above and select Analyse URLs.")
        return
    st.divider()
    worst = max(out["urls"], key=lambda f: f["score"])
    ui.kpi_grid([("URLs analysed", str(len(out["urls"])), "offline"),
                 ("Highest risk", str(worst["score"]), worst["severity"]),
                 ("Flagged (score >= 25)", str(sum(f["score"] >= 25 for f in out["urls"])), "need attention")])
    df = pd.DataFrame([{"URL (defanged)": f["defanged"], "Score": f["score"], "Severity": f["severity"],
                        "Indicators": "; ".join(x["label"] for x in f["flags"])} for f in out["urls"]])
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
    for f in out["urls"]:
        with st.expander(f"{f['defanged'][:100]} - {f['severity']} ({f['score']})"):
            for fl in f["flags"] or []:
                st.markdown(f"- **{fl['label']}** (+{fl['points']}) {fl['detail'] and '- ' + fl['detail']}")
            if not f["flags"]:
                st.write("No indicators triggered. This does not prove the URL is safe.")
    if out["external"] is not None:
        st.markdown("**External reputation (opt-in)**")
        rows = [{"URL (defanged)": next((x["defanged"] for x in out["urls"] if x["url"] == u), u), **r}
                for u, rs in out["external"].items() for r in rs]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("'unknown' = the provider has no record; it does not mean the URL is safe.")


render()
