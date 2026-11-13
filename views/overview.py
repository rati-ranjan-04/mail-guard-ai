"""Overview dashboard."""

import pandas as pd
import streamlit as st

from mailguard import charts, external, ui


def render():
    ui.hero("Overview", "Email security dashboard",
            "Spam classification with a Multinomial Naive Bayes model, plus transparent phishing, URL, sender and "
            "authentication indicators combined into an explainable 0-100 risk score.")
    hist = ui.get_history()
    df = hist.list(limit=2000)
    ev = ui.load_evaluation()

    if df.empty:
        ui.kpi_grid([("Total scans", "0", "no scans yet"), ("High / Critical", "0", "-"),
                     ("Average risk", "-", "-"), ("Last scan", "-", "-")])
        st.info("No scan history yet. Start with the Email scanner (try a synthetic example), "
                "the URL scanner, or Batch analysis.")
        cols = st.columns(3)
        cols[0].page_link("views/email_scanner.py", label="Scan an email", icon=":material/mail:")
        cols[1].page_link("views/url_scanner.py", label="Scan URLs", icon=":material/link:")
        cols[2].page_link("views/batch.py", label="Batch analysis", icon=":material/table_view:")
    else:
        high = int(df["severity"].isin(["High", "Critical"]).sum())
        last = df["ts"].max()
        ui.kpi_grid([("Total scans", f"{len(df):,}", "stored on this machine"),
                     ("High / Critical", f"{high:,}", f"{high / len(df):.1%} of scans"),
                     ("Average risk", f"{df['risk'].mean():.1f}", "out of 100"),
                     ("Last scan", last.strftime("%d %b %H:%M"), "UTC")])
        l, r = st.columns(2)
        with l:
            st.markdown("**Scans by severity**")
            charts.show(charts.severity_bar(df["severity"].value_counts().to_dict()), height=250)
        with r:
            st.markdown("**Risk score over time**")
            charts.show(charts.risk_timeline(df), height=250)
        st.markdown("**Most frequent indicators**")
        ind = (df["indicators"].str.split("; ").explode().replace("", pd.NA).dropna().value_counts().head(8)
               .rename_axis("Indicator").reset_index(name="Scans"))
        if not ind.empty:
            st.dataframe(ind, hide_index=True, width="stretch")

    st.markdown("### Model snapshot")
    if ev is None:
        st.warning("No evaluation report found. Run `python src/train_compare.py` to generate model metrics.")
    else:
        m = ev["models"][ev["primary"]]
        meta = ev["meta"]
        ui.kpi_grid([("F1 score", f"{m['f1']:.3f}", f"95% CI {m['f1_ci95'][0]:.3f}-{m['f1_ci95'][1]:.3f}"),
                     ("Precision / Recall", f"{m['precision']:.3f} / {m['recall']:.3f}", "held-out test set"),
                     ("ROC-AUC", f"{m['roc_auc']:.4f}", f"PR-AUC {m['pr_auc']:.4f}"),
                     ("Model version", meta["version"], f"{meta['dataset']['test_rows_leak_free']:,} leak-free test rows")])
        st.caption("Measured on one spam/ham dataset and one split. This does not guarantee performance on modern or "
                   "real-world mail, and says nothing about phishing detection accuracy. See Analytics & model.")

    st.markdown("### Detection coverage")
    a, b, c = st.columns(3)
    a.success("**Implemented locally**\n\nNaive Bayes spam model, term-level explanations, URL / sender / header / "
              "phishing heuristics, risk score, PDF reports.", icon=":material/check_circle:")
    prov = external.configured_providers()
    b.warning("**Optional external checks**\n\nURL reputation via Google Safe Browsing / VirusTotal. "
              + ("Configured: " + ", ".join(prov) if prov else "Not configured (offline mode)."),
              icon=":material/cloud:")
    c.error("**Not supported**\n\nURL fetching or sandboxing, attachment malware scanning, live DNS or cryptographic "
            "DKIM verification, sender-reputation feeds.", icon=":material/block:")


render()
