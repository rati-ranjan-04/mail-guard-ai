"""PDF security reports (ReportLab). Reports never include the email body or subject."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEV_COLORS = {"Low": "#1a7f4b", "Medium": "#b7791f", "High": "#c2410c", "Critical": "#b91c1c"}
NAVY = colors.HexColor("#0b1b3a")

DISCLAIMER = (
    "Mail Guard AI is a decision-support tool. The spam probability comes from a Multinomial Naive Bayes "
    "model trained on a spam/ham dataset. URL, sender, header and phishing indicators are rule-based "
    "heuristics that have not been benchmarked on labelled phishing data. SPF/DKIM/DMARC values are "
    "parsed from headers, not independently verified. No URL was visited and no attachment was opened. "
    "A low score does not guarantee a message is safe."
)


def _t(value, limit: int = 300) -> str:
    """Escape for ReportLab Paragraph and drop characters Helvetica cannot render."""
    s = str(value)[:limit].encode("latin-1", "replace").decode("latin-1")
    return escape(s)


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=20, leading=24, alignment=0),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13),
        "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=8, leading=10.5, textColor=colors.HexColor("#444444")),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontSize=8.5, leading=11),
    }


def _table(data, widths, header=True):
    st = _styles()["cell"]
    rows = [[Paragraph(_t(c), st) if not hasattr(c, "wrap") else c for c in row] for row in data]
    tbl = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d1e0")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fb")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    tbl.setStyle(TableStyle(style))
    return tbl


def _header_row(labels):
    st = ParagraphStyle("hdr", parent=_styles()["cell"], textColor=colors.white, fontName="Helvetica-Bold")
    return [Paragraph(_t(l), st) for l in labels]


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                             bottomMargin=16 * mm, title=title, author="Mail Guard AI")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 10 * mm, "Mail Guard AI - security report (contains no email body or subject)")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_email_report(result: dict, generated_at: datetime | None = None) -> bytes:
    st = _styles()
    now = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    risk, ml, meta = result["risk"], result["ml"], result["meta"]
    sev_color = SEV_COLORS.get(risk["severity"], "#333333")
    buf = io.BytesIO()
    story = [Paragraph("Email Security Report", st["title"]),
             Paragraph(f"Generated {now} &nbsp;|&nbsp; Model {_t(meta['model_version'])} "
                       f"&nbsp;|&nbsp; Model integrity: {_t(meta['integrity'])} "
                       f"&nbsp;|&nbsp; Content SHA-256: {_t(meta['content_sha256'][:16])}...", st["small"]),
             Spacer(1, 6)]

    box = Table([[Paragraph(f'<font size="30" color="{sev_color}"><b>{int(risk["score"])}</b></font>'
                            f'<font size="11"> / 100</font>', st["body"]),
                  Paragraph(f'<font color="{sev_color}"><b>{_t(risk["severity"])} severity</b></font><br/>'
                            f'Verdict: <b>{_t(risk["verdict"])}</b><br/>'
                            f'Spam probability: {ml["spam_probability"]:.1%} ({_t(ml["probability_source"])})',
                            st["body"])]], colWidths=[40 * mm, 130 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor(sev_color)),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f8fc"))]))
    story += [box, Spacer(1, 6), Paragraph(f"<b>Recommended action:</b> {_t(risk['advice'])}", st["body"])]

    story.append(Paragraph("Risk score breakdown", st["h2"]))
    rows = [_header_row(["Component", "Points", "Max", "Note"])]
    for b in risk["breakdown"]:
        rows.append([b["label"], f'{b["points"]:.1f}', str(b["max_points"]), b["note"]])
    story.append(_table(rows, [72 * mm, 16 * mm, 14 * mm, 68 * mm]))

    story.append(Paragraph("Key findings", st["h2"]))
    reasons = result.get("reasons", [])[:15]
    if reasons:
        rows = [_header_row(["Source", "Finding", "Points", "Detail"])]
        rows += [[r["source"], r["label"], str(r["points"]), r.get("detail", "")] for r in reasons]
        story.append(_table(rows, [28 * mm, 64 * mm, 14 * mm, 64 * mm]))
    else:
        story.append(Paragraph("No rule-based risk indicators were triggered.", st["body"]))

    exp = ml.get("explanation") or {}
    if exp.get("spam_terms") or exp.get("ham_terms"):
        story.append(Paragraph("Why the model scored it this way (Naive Bayes term contributions)", st["h2"]))
        rows = [_header_row(["Pushes toward SPAM", "Weight", "Pushes toward HAM", "Weight"])]
        sp, hm = exp.get("spam_terms", [])[:6], exp.get("ham_terms", [])[:6]
        for i in range(max(len(sp), len(hm))):
            a = sp[i] if i < len(sp) else ("", "")
            b = hm[i] if i < len(hm) else ("", "")
            rows.append([a[0], f"{a[1]:+.2f}" if a[0] else "", b[0], f"{b[1]:+.2f}" if b[0] else ""])
        story.append(_table(rows, [58 * mm, 24 * mm, 58 * mm, 24 * mm]))

    if result.get("urls"):
        story.append(Paragraph("URLs found (defanged - not visited)", st["h2"]))
        rows = [_header_row(["URL", "Score", "Severity"])]
        rows += [[u["defanged"], str(u["score"]), u["severity"]] for u in result["urls"][:25]]
        story.append(_table(rows, [118 * mm, 18 * mm, 34 * mm]))

    s, a = result["sender"], result["auth"]
    story.append(Paragraph("Sender and authentication", st["h2"]))
    if s["assessed"]:
        story.append(Paragraph(f"Sender domain: <b>{_t(s['domain'])}</b> (sub-score {s['score']}/100)", st["body"]))
    else:
        story.append(Paragraph("Sender: not assessed (no From header supplied).", st["body"]))
    if a["assessed"]:
        story.append(Paragraph(f"SPF: <b>{_t(a['spf'])}</b> &nbsp; DKIM: <b>{_t(a['dkim'])}</b> &nbsp; "
                               f"DMARC: <b>{_t(a['dmarc'])}</b> - {_t(a['note'])}", st["body"]))
    else:
        story.append(Paragraph(_t(a["note"]), st["body"]))
    if result.get("attachments"):
        story.append(Paragraph(f"Attachments (not opened): {len(result['attachments'])} - "
                               + _t(", ".join(x["name"] for x in result["attachments"][:8])), st["body"]))
    if result.get("external") is not None:
        flagged = sum(1 for rs in result["external"].values() for r in rs if r["status"] == "flagged")
        story.append(Paragraph(f"External reputation check (opt-in): {flagged} flagged result(s).", st["body"]))

    story += [Spacer(1, 10), Paragraph("Scope and limitations", st["h2"]), Paragraph(_t(DISCLAIMER, 1200), st["small"])]
    _doc(buf, "Email Security Report").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def build_batch_report(summary: dict, rows: list[dict], generated_at: datetime | None = None) -> bytes:
    """summary: {'total','by_severity','by_verdict','avg_risk','model_version', 'truncated'}; rows: top findings."""
    st = _styles()
    now = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    buf = io.BytesIO()
    story = [Paragraph("Batch Email Analysis Report", st["title"]),
             Paragraph(f"Generated {now} | Model {_t(summary.get('model_version', ''))} | "
                       f"{summary['total']} messages | average risk {summary['avg_risk']:.1f}/100", st["small"]),
             Spacer(1, 8), Paragraph("Severity distribution", st["h2"])]
    sev_rows = [_header_row(["Severity", "Messages", "Share"])]
    for sev in ("Critical", "High", "Medium", "Low"):
        n = summary["by_severity"].get(sev, 0)
        sev_rows.append([sev, str(n), f"{n / max(1, summary['total']):.1%}"])
    story.append(_table(sev_rows, [60 * mm, 40 * mm, 40 * mm]))
    story.append(Paragraph("Verdicts", st["h2"]))
    v_rows = [_header_row(["Verdict", "Messages"])] + [[k, str(v)] for k, v in summary["by_verdict"].items()]
    story.append(_table(v_rows, [90 * mm, 40 * mm]))
    if summary.get("evaluation"):
        e = summary["evaluation"]
        story.append(Paragraph("Agreement with supplied labels", st["h2"]))
        story.append(Paragraph(f"Precision {e['precision']:.3f} | Recall {e['recall']:.3f} | F1 {e['f1']:.3f} | "
                               f"n={e['n']} (labels supplied by the uploader; spam = 1)", st["body"]))
    story.append(Paragraph("Highest-risk messages (row number, no content)", st["h2"]))
    t_rows = [_header_row(["Row", "Score", "Severity", "Verdict", "Top indicators"])]
    for r in rows[:30]:
        t_rows.append([str(r["row"]), str(r["risk"]), r["severity"], r["verdict"], r["indicators"]])
    story.append(_table(t_rows, [14 * mm, 14 * mm, 20 * mm, 36 * mm, 86 * mm]))
    if summary.get("truncated"):
        story.append(Paragraph("Note: the upload exceeded the row limit; only the first rows were analysed.", st["small"]))
    story += [Spacer(1, 10), Paragraph("Scope and limitations", st["h2"]), Paragraph(_t(DISCLAIMER, 1200), st["small"])]
    _doc(buf, "Batch Email Analysis Report").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
