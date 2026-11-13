"""Email scanner page."""

import pandas as pd
import streamlit as st

from logger import get_logger
from mailguard import charts, external, ui
from mailguard.samples import SAMPLES
from mailguard.security import (MAX_EML_BYTES, MAX_TEXT_CHARS, ValidationError, clean_input_text,
                                decode_bytes, esc, validate_upload)

logger = get_logger(__name__)
KEY = "email_text"


def _load_sample():
    name = st.session_state.get("sample_choice")
    if name in SAMPLES:
        st.session_state[KEY] = SAMPLES[name]


def _scan(raw: str, use_external: bool, save: bool):
    engine = ui.require_engine()
    with st.status("Scanning email...", expanded=True) as status:
        try:
            st.write("Validating input and parsing message")
            raw = clean_input_text(raw)
            st.write("Running Naive Bayes model, URL, sender, header and phishing checks")
            result = engine.analyze_email(raw, use_external=use_external)
            if save:
                ui.get_history().add(result, "email")
            st.session_state["email_result"] = result
            status.update(label="Scan complete", state="complete", expanded=False)
            st.toast("Scan complete", icon="✅")
        except ValidationError as exc:
            status.update(label="Input rejected", state="error")
            st.error(str(exc))
        except Exception as exc:
            logger.error("Email scan failed: %s", type(exc).__name__)
            status.update(label="Scan failed", state="error")
            st.error("The scan failed unexpectedly. The details were logged without email content.")


def _flags_table(reasons):
    df = pd.DataFrame(reasons)[["source", "label", "points", "detail"]]
    df.columns = ["Source", "Finding", "Points", "Detail"]
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Points": st.column_config.ProgressColumn("Points", min_value=0, max_value=40, format="%d")})


def _results(res: dict):
    ui.result_header(res)
    ml = res["ml"]
    ui.kpi_grid([
        ("Spam probability", f"{ml['spam_probability']:.1%}", ml["probability_source"]),
        ("Classifier label", ml["label"], "Multinomial Naive Bayes"),
        ("URLs found", str(len(res["urls"])), "analysed offline, not opened"),
        ("Attachments", str(len(res["attachments"])), "described only, never opened"),
    ])
    if res["meta"]["truncated"]:
        st.warning("The message was longer than the limit and was truncated before analysis.")
    if res["meta"]["integrity"] != "verified":
        st.warning("Model integrity could not be verified against models/manifest.json. Results are still produced.")

    t1, t2, t3, t4, t5 = st.tabs(["Why this score", "Model explanation", "URLs", "Sender & headers", "Report"])
    with t1:
        charts.show(charts.breakdown_bar(res["risk"]["breakdown"]), height=230)
        for b in res["risk"]["breakdown"]:
            if not b["assessed"]:
                st.caption(f"• {b['label']}: {b['note']} (no points added, not counted as a failure)")
        st.markdown("**Findings**")
        if res["reasons"]:
            _flags_table(res["reasons"])
        else:
            st.success("No rule-based risk indicators were triggered.")
    with t2:
        ex = ml["explanation"]
        st.caption("Exact decomposition of the Naive Bayes log-odds: each term's weight is "
                   "TF-IDF value x (log P(term|spam) - log P(term|ham)).")
        if ex["spam_terms"] or ex["ham_terms"]:
            charts.show(charts.contributions_bar(ex["spam_terms"], ex["ham_terms"]), height=max(240, 24 * (len(ex["spam_terms"]) + len(ex["ham_terms"]))))
        if ex["note"]:
            st.info(ex["note"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Known vocabulary terms", ex["n_known_terms"])
        c2.metric("Prior log-odds", f"{ex['prior_log_odds']:+.2f}")
        c3.metric("Total log-odds", f"{ex['log_odds']:+.2f}")
        st.caption(f"Raw model score: {ml['raw_probability']:.1%}. Displayed probability: {ml['spam_probability']:.1%} "
                   f"({ml['probability_source']}).")
    with t3:
        ui.not_visited_note()
        if not res["urls"]:
            st.info("No URLs found in the message body.")
        for u in res["urls"]:
            with st.expander(f"{u['defanged'][:90]}  -  {u['severity']} ({u['score']})"):
                if u["flags"]:
                    for f in u["flags"]:
                        st.markdown(f"- **{f['label']}** (+{f['points']}) {f['detail'] and '- ' + f['detail']}")
                else:
                    st.write("No risk indicators triggered by the offline heuristics.")
        if res["external"] is not None:
            st.markdown("**External reputation results**")
            for url, rs in res["external"].items():
                for r in rs:
                    st.write(f"{r['provider']}: **{r['status']}** - {r['detail']}")
    with t4:
        s, a = res["sender"], res["auth"]
        st.markdown("**Sender**")
        if s["assessed"]:
            st.write(f"Display name: `{s['display_name'] or '-'}`  |  Address: `{s['address']}`  |  Reply-To domain: `{s['reply_to_domain'] or '-'}`")
            for f in s["flags"]:
                st.markdown(f"- **{f['label']}** (+{f['points']}) {f['detail'] and '- ' + f['detail']}")
            if not s["flags"]:
                st.write("No sender risk indicators triggered.")
        else:
            st.info("No From header supplied (pasted body only), so sender checks were skipped.")
        st.markdown("**SPF / DKIM / DMARC (parsed from headers)**")
        if a["assessed"]:
            c1, c2, c3 = st.columns(3)
            c1.metric("SPF", a["spf"].upper())
            c2.metric("DKIM", a["dkim"].upper())
            c3.metric("DMARC", a["dmarc"].upper())
            for f in a["flags"]:
                st.markdown(f"- {f['label']} (+{f['points']}) {f['detail'] and '- ' + f['detail']}")
        st.caption(a["note"])
        if res["attachments"]:
            st.markdown("**Attachments (never opened)**")
            st.dataframe(pd.DataFrame(res["attachments"]), hide_index=True, width="stretch")
    with t5:
        st.write("Download a PDF with the score, breakdown, findings and limitations. "
                 "It contains no email body or subject.")
        try:
            from mailguard.report_pdf import build_email_report  # lazy: PDF needs reportlab
            pdf = build_email_report(res)
            st.download_button("Download PDF report", pdf, file_name="mail-guard-report.pdf",
                               mime="application/pdf", icon=":material/picture_as_pdf:", on_click="ignore", type="primary")
        except ImportError:
            st.warning("PDF reports need the 'reportlab' package. Run `pip install -r requirements.txt` and restart the app.")
        except Exception as exc:
            logger.error("PDF build failed: %s", type(exc).__name__)
            st.error("The PDF report could not be generated.")


def render():
    ui.hero("Email scanner", "Analyse a message",
            "Paste a message (with or without headers) or upload an .eml file. Spam probability, phishing language, "
            "URLs, sender and authentication results are combined into a 0-100 risk score with reasons.")
    ui.not_visited_note()
    st.write("")
    tab_paste, tab_file = st.tabs(["Paste text / raw message", "Upload .eml"])
    raw_to_scan = None
    with tab_paste:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.selectbox("Load a synthetic example", list(SAMPLES), key="sample_choice", label_visibility="collapsed")
        with c2:
            st.button("Load example", on_click=_load_sample, width="stretch", icon=":material/science:")
        st.text_area("Email content", key=KEY, height=250, max_chars=MAX_TEXT_CHARS,
                     placeholder="Paste the full raw message including headers for sender and SPF/DKIM/DMARC checks, "
                                 "or just the subject and body.")
        paste_clicked = False
    with tab_file:
        up = st.file_uploader("Choose an .eml file (max 2 MB). It is parsed as text only.", type=["eml"],
                              accept_multiple_files=False)
        file_clicked = st.button("Scan uploaded file", disabled=up is None, icon=":material/upload_file:")

    with st.expander("Options"):
        providers = external.configured_providers()
        use_ext = st.checkbox("Also run external URL reputation check", value=False, disabled=not providers,
                              help="Sends only the URL strings to the configured provider. Off by default.")
        if providers:
            st.caption("Configured: " + ", ".join(providers) + ". URLs will leave this machine if enabled.")
        else:
            st.caption("No API keys configured (see README), so the app runs fully offline.")
        save = st.checkbox("Save scan to history (stores a hash and scores, never the email text)", value=True)

    if st.button("Scan email", type="primary", icon=":material/policy:", key="scan_paste"):
        raw_to_scan = st.session_state.get(KEY, "")
    if file_clicked and up is not None:
        try:
            data = validate_upload(up.name, up.getvalue(), (".eml",), MAX_EML_BYTES)
            raw_to_scan = decode_bytes(data)
        except ValidationError as exc:
            st.error(str(exc))
    if raw_to_scan is not None:
        _scan(raw_to_scan, use_ext, save)

    res = st.session_state.get("email_result")
    if res:
        st.divider()
        _results(res)
    else:
        st.info("Results will appear here after a scan. Try a synthetic example to see how it works.")


render()
