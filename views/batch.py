"""Batch CSV analysis page."""

import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score

from logger import get_logger
from mailguard import charts, ui
from mailguard.samples import SAMPLE_BATCH_CSV
from mailguard.security import (MAX_BATCH_ROWS, MAX_CSV_BYTES, ValidationError, read_csv_safely,
                                sanitize_dataframe_for_export, validate_upload)

logger = get_logger(__name__)
TEXT_GUESS = ("text", "body", "message", "email", "content", "mail")
SENDER_GUESS = ("sender", "from", "from_address", "from_email")
SUBJECT_GUESS = ("subject", "title")
POS = {"1", "spam", "true", "yes", "phish", "phishing"}
NEG = {"0", "ham", "false", "no", "legit", "legitimate", "not spam"}


def _guess(cols, options):
    low = {c.lower(): c for c in cols}
    for o in options:
        if o in low:
            return low[o]
    return None


def _labels(series):
    vals = series.str.strip().str.lower()
    if not vals.isin(POS | NEG).all():
        return None
    return vals.isin(POS).astype(int)


def _run(df, text_col, sender_col, subject_col, label_col, truncated):
    engine = ui.require_engine()
    rows = [{"text": r[text_col], "sender": r[sender_col] if sender_col else "",
             "subject": r[subject_col] if subject_col else ""} for _, r in df.iterrows()]
    bar = st.progress(0.0, text="Analysing messages...")
    results = engine.analyze_batch(rows, progress=lambda f: bar.progress(f, text=f"Analysing messages... {f:.0%}"))
    bar.empty()
    out = pd.DataFrame({
        "Row": range(1, len(results) + 1),
        "Verdict": [r["risk"]["verdict"] for r in results],
        "Risk": [r["risk"]["score"] for r in results],
        "Severity": [r["risk"]["severity"] for r in results],
        "Spam probability": [round(r["ml"]["spam_probability"], 4) for r in results],
        "Classifier label": [r["ml"]["label"] for r in results],
        "URLs": [len(r["urls"]) for r in results],
        "Sender domain": [r["sender"]["domain"] for r in results],
        "Top indicators": ["; ".join(x["label"] for x in r["reasons"][:3]) for r in results],
    })
    summary = {"total": len(out), "avg_risk": float(out["Risk"].mean()), "model_version": engine.model_version,
               "by_severity": out["Severity"].value_counts().to_dict(),
               "by_verdict": out["Verdict"].value_counts().to_dict(), "truncated": truncated}
    if label_col:
        y = _labels(df[label_col])
        if y is None:
            st.warning("The label column has values other than spam/ham/0/1, so label agreement was skipped.")
        else:
            pred = (out["Spam probability"] >= 0.5).astype(int)
            summary["evaluation"] = {"precision": float(precision_score(y, pred, zero_division=0)),
                                     "recall": float(recall_score(y, pred, zero_division=0)),
                                     "f1": float(f1_score(y, pred, zero_division=0)), "n": int(len(y))}
            out["Supplied label"] = y.map({1: "spam", 0: "ham"}).values
    return out, summary, results


def render():
    ui.hero("Batch analysis", "Scan many emails from a CSV",
            f"Upload a CSV (max {MAX_CSV_BYTES // 1_000_000} MB, {MAX_BATCH_ROWS:,} rows). Each row is analysed locally; "
            "the exported results contain row numbers and scores, not email text.")
    c1, c2 = st.columns([3, 1])
    with c1:
        up = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
    with c2:
        st.write("")
        st.download_button("Sample CSV", SAMPLE_BATCH_CSV, "sample_batch.csv", "text/csv", on_click="ignore",
                           icon=":material/download:", width="stretch")
    if up is None:
        st.info("Upload a CSV with a column containing email text. Optional columns: sender, subject, and a label column "
                "(spam/ham or 1/0) to measure how well the model agrees with your labels.")
    else:
        try:
            data = validate_upload(up.name, up.getvalue(), (".csv",), MAX_CSV_BYTES)
            df, truncated = read_csv_safely(data)
        except ValidationError as exc:
            st.error(str(exc))
            return
        if truncated:
            st.warning(f"Only the first {MAX_BATCH_ROWS:,} rows will be analysed.")
        st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        cols = list(df.columns)
        none = "(none)"
        a, b, c, d = st.columns(4)
        tg, sg, ug = _guess(cols, TEXT_GUESS), _guess(cols, SENDER_GUESS), _guess(cols, SUBJECT_GUESS)
        text_col = a.selectbox("Email text column", cols, index=cols.index(tg) if tg else 0)
        sender_col = b.selectbox("Sender column", [none] + cols, index=([none] + cols).index(sg) if sg else 0)
        subject_col = c.selectbox("Subject column", [none] + cols, index=([none] + cols).index(ug) if ug else 0)
        lg = _guess(cols, ("label", "spam", "class", "target"))
        label_col = d.selectbox("Label column (optional)", [none] + cols, index=([none] + cols).index(lg) if lg and lg != text_col else 0)
        with st.expander("Preview (first 5 rows, truncated cells)"):
            st.dataframe(df.head(5).map(lambda v: v[:80]), width="stretch")
        save = st.checkbox("Save each result to scan history (hash + scores only)", value=False)
        if st.button("Run batch analysis", type="primary", icon=":material/playlist_add_check:"):
            try:
                out, summary, results = _run(df, text_col, None if sender_col == none else sender_col,
                                             None if subject_col == none else subject_col,
                                             None if label_col == none else label_col, truncated)
                if save:
                    hist = ui.get_history()
                    for r in results:
                        hist.add(r, "batch")
                st.session_state["batch"] = (out, summary)
                st.toast(f"Analysed {len(out)} messages", icon="✅")
            except Exception as exc:
                logger.error("Batch failed: %s", type(exc).__name__)
                st.error("The batch analysis failed unexpectedly. Check that the selected text column contains email text.")

    state = st.session_state.get("batch")
    if not state:
        return
    out, summary = state
    st.divider()
    flagged = int(out["Severity"].isin(["High", "Critical"]).sum())
    ui.kpi_grid([("Messages", f"{summary['total']:,}", "analysed"),
                 ("High / Critical", f"{flagged:,}", f"{flagged / max(1, summary['total']):.1%} of batch"),
                 ("Average risk", f"{summary['avg_risk']:.1f}", "out of 100"),
                 ("Model", summary["model_version"], "Naive Bayes primary")])
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Severity distribution**")
        charts.show(charts.severity_bar(summary["by_severity"]), height=240)
    with right:
        st.markdown("**Verdicts**")
        st.dataframe(pd.DataFrame(list(summary["by_verdict"].items()), columns=["Verdict", "Messages"]),
                     hide_index=True, width="stretch")
        if "evaluation" in summary:
            e = summary["evaluation"]
            st.markdown("**Agreement with your labels (classifier label vs supplied label)**")
            st.write(f"Precision **{e['precision']:.3f}** | Recall **{e['recall']:.3f}** | F1 **{e['f1']:.3f}** | n={e['n']}")
            st.caption("Measures the Naive Bayes label only, against labels you supplied. Not a security guarantee.")
    sev = st.multiselect("Filter by severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    view = out[out["Severity"].isin(sev)].sort_values("Risk", ascending=False)
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
                                "Spam probability": st.column_config.NumberColumn(format="%.3f")})
    d1, d2 = st.columns(2)
    d1.download_button("Download results (CSV)", sanitize_dataframe_for_export(view).to_csv(index=False),
                       "mail-guard-batch-results.csv", "text/csv", on_click="ignore", icon=":material/table:",
                       width="stretch")
    top = out.sort_values("Risk", ascending=False).head(30)
    rows = [{"row": int(r["Row"]), "risk": int(r["Risk"]), "severity": r["Severity"], "verdict": r["Verdict"],
             "indicators": r["Top indicators"]} for _, r in top.iterrows()]
    try:
        from mailguard.report_pdf import build_batch_report  # lazy: PDF needs reportlab
        pdf = build_batch_report(summary, rows)
    except ImportError:
        d2.warning("PDF reports need 'reportlab': run `pip install -r requirements.txt` and restart.")
        return
    d2.download_button("Download PDF summary", pdf, "mail-guard-batch-report.pdf",
                       "application/pdf", on_click="ignore", icon=":material/picture_as_pdf:", type="primary",
                       width="stretch")


render()
