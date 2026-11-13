"""Analytics and model performance page. All numbers come from reports/evaluation.json."""

import pandas as pd
import streamlit as st

import config
from mailguard import charts, ui


def _pct(x):
    return f"{x:.2%}" if x is not None else "-"


def render():
    ui.hero("Analytics & model", "Model performance and evaluation",
            "Metrics are computed by src/train_compare.py on a held-out test set and stored in reports/evaluation.json. "
            "Nothing on this page is hard-coded.")
    ev = ui.load_evaluation()
    if ev is None:
        st.warning("No evaluation report found. Generate it with:")
        st.code("python src/train_compare.py", language="bash")
        return
    meta, models, primary = ev["meta"], ev["models"], ev["primary"]
    d = meta["dataset"]

    engine = ui.get_engine() if (config.MODEL_PATH).exists() else None
    ui.kpi_grid([("Active model", meta["version"], f"scikit-learn {meta['libs']['scikit-learn']}"),
                 ("Integrity", engine.integrity_status if engine else "n/a", "SHA-256 vs manifest"),
                 ("Probabilities", "Calibrated" if engine and engine.calibrator else "Raw", "Platt scaling on NB log-odds"),
                 ("Training rows", f"{d['train_rows']:,}", f"seed {meta['seed']}, stratified 80/20")])
    st.caption(f"Dataset `{d['file']}` sha256 `{d['sha256'][:16]}...` | {d['rows_after_cleaning']:,} rows after cleaning | "
               f"spam rate {d['spam_rate']:.1%} | test set: {d['test_rows_leak_free']:,} rows after removing "
               f"{d['removed_overlap']} that also appear in the training set (the original code de-duplicated "
               "before cleaning, which left such duplicates).")

    tabs = st.tabs(["Model comparison", "ROC & PR", "Confusion matrix", "Calibration", "Thresholds", "Errors (FP / FN)",
                    "Top features", "Reproducibility"])
    order = sorted(models, key=lambda k: -models[k]["f1"])

    with tabs[0]:
        rows = [{"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"], "Recall": v["recall"],
                 "F1": v["f1"], "F1 95% CI": f"{v['f1_ci95'][0]:.3f}-{v['f1_ci95'][1]:.3f}", "ROC-AUC": v["roc_auc"],
                 "PR-AUC": v["pr_auc"], "Brier": v["brier"], "FP": v["confusion"]["fp"], "FN": v["confusion"]["fn"]}
                for k, v in ((k, models[k]) for k in order)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                    ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier")})
        metric = st.segmented_control("Metric", ["F1", "ROC-AUC", "PR-AUC", "Precision", "Recall"], default="F1")
        key = {"F1": "f1", "ROC-AUC": "roc_auc", "PR-AUC": "pr_auc", "Precision": "precision", "Recall": "recall"}[metric or "F1"]
        charts.show(charts.metric_bars(models, key, metric or "F1", order, primary), height=260)
        best = order[0]
        st.info(f"**Reading this honestly.** {primary} is the primary classifier (F1 {models[primary]['f1']:.3f}). "
                f"Highest F1 on this split: **{best}** ({models[best]['f1']:.3f}). The F1 confidence intervals of the "
                "top models overlap, so on ~1,000 test messages the differences between them are not statistically "
                "meaningful. Naive Bayes is kept because it is fast, tiny, and exactly explainable. "
                "Baselines show what 'no learning' scores.")

    with tabs[1]:
        names = [primary] + [n for n in ("Logistic Regression", "Linear SVM", "Complement Naive Bayes") if n in models][:2]
        l, r = st.columns(2)
        with l:
            st.markdown("**ROC curve**")
            charts.show(charts.curves_chart(ev["curves"], "roc", names, primary), height=300)
        with r:
            st.markdown("**Precision-Recall curve**")
            charts.show(charts.curves_chart(ev["curves"], "pr", names, primary), height=300)
        st.caption("With a 25% spam rate, the Precision-Recall curve is the more informative view for the minority class. "
                   "Linear SVM scores are decision-function values (not probabilities), which is fine for ROC/PR.")

    with tabs[2]:
        which = st.selectbox("Model", order, index=order.index(primary))
        cm = models[which]["confusion"]
        l, r = st.columns([1, 1])
        with l:
            charts.show(charts.confusion_chart(cm), height=280)
        with r:
            m = models[which]
            ui.kpi_grid([("Precision", _pct(m["precision"]), "of flagged spam, how much was spam"),
                         ("Recall", _pct(m["recall"]), "of all spam, how much was caught"),
                         ("False positives", str(cm["fp"]), "ham flagged as spam"),
                         ("False negatives", str(cm["fn"]), "spam that got through")])
        st.caption("Rows = actual class, columns = predicted class. Threshold 0.5 on the model's own probability.")

    cal = ev["calibration"]
    with tabs[3]:
        l, r = st.columns([3, 2])
        with l:
            charts.show(charts.reliability_chart(cal["reliability_raw"], cal["reliability_calibrated"]), height=320)
        with r:
            st.metric("Brier score (lower is better)", f"{cal['brier_calibrated']:.4f}", f"{cal['brier_calibrated'] - cal['brier_raw']:+.4f} vs raw")
            st.metric("Expected calibration error", f"{cal['ece_calibrated']:.4f}", f"{cal['ece_calibrated'] - cal['ece_raw']:+.4f} vs raw")
            st.caption(f"Method: {cal['method']}, fitted on out-of-fold training predictions only. "
                       f"Raw and calibrated labels agree on {cal['label_agreement_raw_vs_calibrated']:.1%} of test messages.")
        st.caption("Points on the dashed diagonal mean 'when the model says 70%, about 70% really are spam'. "
                   "The dataset is easy, so even raw NB scores are fairly well calibrated; calibration mainly helps in the middle range.")

    with tabs[4]:
        th = pd.DataFrame(ev["thresholds"])
        st.dataframe(th, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("precision", "recall", "f1")})
        st.caption("Lower thresholds catch more spam (fewer false negatives) but flag more legitimate mail (more false positives). "
                   "The app uses 0.5 by default; choose a threshold according to the cost of each error type.")

    with tabs[5]:
        errs = ev["errors"]
        c = errs["counts"]
        st.write(f"On the leak-free test set the calibrated primary model made **{c['false_positives']} false positives** "
                 f"(legitimate flagged as spam) and **{c['false_negatives']} false negatives** (spam missed). "
                 "Snippets are from the public training corpus and are shortened.")
        for title, key in (("False positives (ham predicted as spam)", "false_positives"),
                           ("False negatives (spam predicted as ham)", "false_negatives")):
            st.markdown(f"**{title}**")
            if errs[key]:
                st.dataframe(pd.DataFrame([{"Spam prob.": e["spam_probability"], "Snippet": e["snippet"],
                                            "Pushed toward spam": ", ".join(e["spam_terms"]),
                                            "Pushed toward ham": ", ".join(e["ham_terms"])} for e in errs[key]]),
                             hide_index=True, width="stretch",
                             column_config={"Spam prob.": st.column_config.NumberColumn(format="%.3f")})
            else:
                st.success("None in this test set.")
        st.caption("Pattern to look for: false positives are often business/technical text containing promotional wording; "
                   "false negatives are often very short or image-style spam with few known words.")

    with tabs[6]:
        l, r = st.columns(2)
        tf = ev["top_features"]
        l.markdown("**Strongest spam terms**")
        l.dataframe(pd.DataFrame(tf["spam"], columns=["Term", "log P(spam)/P(ham)"]), hide_index=True, width="stretch")
        r.markdown("**Strongest ham terms**")
        r.dataframe(pd.DataFrame(tf["ham"], columns=["Term", "log P(spam)/P(ham)"]), hide_index=True, width="stretch")
        st.caption("Terms reflect the training corpus (corporate e-mail as ham), so words like 'enron' or 'thanks' signal ham there "
                   "but would not generalise to every organisation.")

    with tabs[7]:
        st.markdown("**How to reproduce every number on this page**")
        st.code("python src/train_compare.py", language="bash")
        st.json({"version": meta["version"], "seed": meta["seed"], "split": meta["split"], "libs": meta["libs"],
                 "dataset_sha256": d["sha256"], "created_at": meta["created_at"],
                 "shipped_model_vs_retrained_agreement": ev["parity"]["agreement_with_retrained_on_test"],
                 "original_split_metrics_shipped_model": {k: round(ev["original_split_shipped_model"][k], 4)
                                                          for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}},
                expanded=False)
        st.caption("`original_split_metrics_shipped_model` reproduces the numbers reported in the original README "
                   "(accuracy 98.95%, F1 97.80%). The leak-free numbers above are slightly different because overlapping "
                   "test rows were removed.")


render()
