"""Altair chart builders. Colour roles: severity palette (status, with labels) and a validated
blue/orange pair for ham/spam. Every chart has direct labels or tooltips plus a data table elsewhere."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .ui import tokens

SEVERITIES = ["Low", "Medium", "High", "Critical"]


def _cfg(chart, height=260):
    c = tokens()
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=c["track"], domainColor=c["border"], tickColor=c["border"],
                            labelColor=c["muted"], titleColor=c["muted"], labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=c["muted"], titleColor=c["muted"])
            .configure(background="transparent"))


def show(chart, height=260):
    import streamlit as st
    st.altair_chart(_cfg(chart, height), width="stretch", theme=None)


def severity_bar(counts: dict):
    c = tokens()
    df = pd.DataFrame({"Severity": SEVERITIES, "Scans": [int(counts.get(s, 0)) for s in SEVERITIES]})
    base = alt.Chart(df).encode(
        x=alt.X("Severity:N", sort=SEVERITIES, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("Scans:Q", axis=alt.Axis(tickMinStep=1, title="Scans")),
        color=alt.Color("Severity:N", scale=alt.Scale(domain=SEVERITIES, range=[c[s] for s in SEVERITIES]), legend=None),
        tooltip=["Severity", "Scans"])
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=46)
    text = base.mark_text(dy=-8, color=c["text"], fontWeight="bold").encode(text="Scans:Q")
    return bars + text


def risk_timeline(df: pd.DataFrame):
    c = tokens()
    d = df.sort_values("ts").tail(100).copy()
    d["Scan"] = range(1, len(d) + 1)
    line = alt.Chart(d).mark_line(color=c["accent"], strokeWidth=2).encode(
        x=alt.X("Scan:Q", axis=alt.Axis(tickMinStep=1, title="Scan # (most recent 100)")),
        y=alt.Y("risk:Q", scale=alt.Scale(domain=[0, 100]), title="Risk score"))
    pts = alt.Chart(d).mark_circle(size=70, stroke=c["card"], strokeWidth=2).encode(
        x="Scan:Q", y="risk:Q",
        color=alt.Color("severity:N", scale=alt.Scale(domain=SEVERITIES, range=[c[s] for s in SEVERITIES]),
                        legend=alt.Legend(title="Severity", orient="bottom")),
        tooltip=["ts:T", "kind", "verdict", "risk", "severity"])
    return line + pts


def contributions_bar(spam_terms, ham_terms):
    c = tokens()
    rows = [{"Term": t, "Contribution": v, "Direction": "Toward spam"} for t, v in spam_terms]
    rows += [{"Term": t, "Contribution": v, "Direction": "Toward ham"} for t, v in ham_terms]
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Contribution:Q", title="Contribution to spam log-odds  (<- ham | spam ->)"),
        y=alt.Y("Term:N", sort=alt.EncodingSortField("Contribution", order="descending"), title=None),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Toward spam", "Toward ham"], range=[c["spam"], c["ham"]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Term", alt.Tooltip("Contribution:Q", format="+.3f"), "Direction"])


def breakdown_bar(breakdown: list):
    c = tokens()
    df = pd.DataFrame([{"Component": b["label"], "Points": b["points"], "Max": b["max_points"], "assessed": b["assessed"]}
                       for b in breakdown])
    y = alt.Y("Component:N", sort=None, title=None, axis=alt.Axis(labelLimit=260))
    back = alt.Chart(df).mark_bar(color=c["track"], cornerRadiusEnd=4, size=14).encode(
        x=alt.X("Max:Q", title="Points (filled = earned, track = maximum)", scale=alt.Scale(domain=[0, 40])), y=y)
    front = alt.Chart(df).mark_bar(color=c["accent"], cornerRadiusEnd=4, size=14).encode(
        x="Points:Q", y=y, tooltip=["Component", "Points", "Max", "assessed"])
    return back + front


def curves_chart(curves: dict, kind: str, names: list, highlight: str):
    c = tokens()
    rows = []
    for n in names:
        pts = curves[n][kind]
        rows += [{"Model": n, "x": x, "y": y} for x, y in zip(pts["x"], pts["y"])]
    df = pd.DataFrame(rows)
    palette = [c["ham"], c["spam"], "#1baf7a", "#eda100"]
    xt, yt = ("False positive rate", "True positive rate") if kind == "roc" else ("Recall", "Precision")
    dom = [0.0, 1.0]
    ch = alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after" if kind == "pr" else "linear").encode(
        x=alt.X("x:Q", title=xt, scale=alt.Scale(domain=dom)),
        y=alt.Y("y:Q", title=yt, scale=alt.Scale(domain=[0.0, 1.0] if kind == "roc" else [0.5, 1.0])),
        color=alt.Color("Model:N", scale=alt.Scale(domain=names, range=palette[:len(names)]),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.datum.Model == highlight, alt.value([1, 0]), alt.value([5, 3])),
        tooltip=["Model", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")])
    return ch


def confusion_chart(cm: dict):
    c = tokens()
    df = pd.DataFrame([
        {"Actual": "Ham", "Predicted": "Ham", "Count": cm["tn"], "Cell": "True negative"},
        {"Actual": "Ham", "Predicted": "Spam", "Count": cm["fp"], "Cell": "False positive"},
        {"Actual": "Spam", "Predicted": "Ham", "Count": cm["fn"], "Cell": "False negative"},
        {"Actual": "Spam", "Predicted": "Spam", "Count": cm["tp"], "Cell": "True positive"},
    ])
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=["Ham", "Spam"], axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Actual:N", sort=["Ham", "Spam"]))
    rect = base.mark_rect(cornerRadius=6, stroke=c["card"], strokeWidth=3).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(range=[c["card2"], c["accent"]], type="sqrt"), legend=None),
        tooltip=["Cell", "Count"])
    txt = base.mark_text(fontSize=18, fontWeight="bold").encode(
        text=alt.Text("Count:Q"), color=alt.condition(alt.datum.Count > cm["tn"] / 2, alt.value("#ffffff"), alt.value(c["text"])))
    lab = base.mark_text(dy=20, fontSize=10).encode(text="Cell:N", color=alt.condition(
        alt.datum.Count > cm["tn"] / 2, alt.value("#e6eeff"), alt.value(c["muted"])))
    return rect + txt + lab


def reliability_chart(raw, cal):
    c = tokens()
    rows = [{"Series": "Raw NB score", "Predicted": p, "Observed": a, "n": n} for p, a, n in raw]
    rows += [{"Series": "Calibrated", "Predicted": p, "Observed": a, "n": n} for p, a, n in cal]
    df = pd.DataFrame(rows)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=c["muted"], strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    lines = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
        x=alt.X("Predicted:Q", title="Mean predicted spam probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Observed:Q", title="Observed spam fraction", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Raw NB score", "Calibrated"], range=[c["spam"], c["ham"]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Series", alt.Tooltip("Predicted:Q", format=".3f"), alt.Tooltip("Observed:Q", format=".3f"), "n"])
    return diag + lines


def metric_bars(models: dict, metric: str, label: str, order: list, primary: str):
    c = tokens()
    df = pd.DataFrame([{"Model": m, label: models[m][metric] or 0.0,
                        "Role": "Primary" if m == primary else "Comparison"} for m in order])
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(f"{label}:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Role:N", scale=alt.Scale(domain=["Primary", "Comparison"], range=[c["accent"], c["muted"]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", alt.Tooltip(f"{label}:Q", format=".4f")])
    return base.mark_bar(cornerRadiusEnd=4, size=16) + base.mark_text(align="left", dx=4, color=c["text"], fontSize=11).encode(
        text=alt.Text(f"{label}:Q", format=".3f"))
