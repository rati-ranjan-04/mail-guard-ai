"""Scan history page."""

import pandas as pd
import streamlit as st

from mailguard import charts, ui
from mailguard.security import sanitize_dataframe_for_export


def render():
    ui.hero("Scan history", "Past scans on this machine",
            "History stores timestamps, content hashes, scores, severity, sender domain and generic indicator labels. "
            "It never stores email subjects, bodies or URL paths.")
    hist = ui.get_history()
    df = hist.list(limit=2000)
    if df.empty:
        st.info("No scans yet. Run a scan from the Email scanner or URL scanner and it will appear here.")
        return
    f1, f2, f3 = st.columns([2, 1, 2])
    sev = f1.multiselect("Severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    kinds = f2.multiselect("Type", sorted(df["kind"].unique()), default=sorted(df["kind"].unique()))
    q = f3.text_input("Search sender domain / indicators", placeholder="e.g. paypal, DMARC")
    view = df[df["severity"].isin(sev) & df["kind"].isin(kinds)]
    if q.strip():
        mask = view["sender_domain"].str.contains(q, case=False, regex=False) | view["indicators"].str.contains(q, case=False, regex=False)
        view = view[mask]
    ui.kpi_grid([("Shown", f"{len(view):,}", f"of {hist.count():,} stored"),
                 ("Average risk", f"{view['risk'].mean():.1f}" if len(view) else "-", "filtered"),
                 ("High / Critical", f"{int(view['severity'].isin(['High', 'Critical']).sum()):,}", "filtered")])
    shown = view.drop(columns=["content_sha256"]).rename(columns={
        "ts": "Time (UTC)", "kind": "Type", "verdict": "Verdict", "risk": "Risk", "severity": "Severity",
        "spam_probability": "Spam prob.", "sender_domain": "Sender domain", "n_urls": "URLs",
        "indicators": "Indicators", "model_version": "Model", "length": "Length", "id": "ID"})
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
                                "Spam prob.": st.column_config.NumberColumn(format="%.3f")})
    c1, c2 = st.columns(2)
    c1.download_button("Export history (CSV)", sanitize_dataframe_for_export(shown).to_csv(index=False),
                       "mail-guard-history.csv", "text/csv", on_click="ignore", icon=":material/download:", width="stretch")
    with c2.popover("Clear all history", icon=":material/delete:", width="stretch"):
        st.warning("This permanently deletes every stored scan record on this machine.")
        if st.button("Yes, delete all history", type="primary"):
            hist.clear()
            st.toast("History cleared", icon="🗑️")
            st.rerun()


render()
