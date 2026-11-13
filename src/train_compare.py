"""Reproducible training, model comparison, calibration and evaluation.

Run:  python src/train_compare.py            (does NOT overwrite the shipped model)
      python src/train_compare.py --promote  (also replace models/spam_classifier_pipeline.joblib)

What it does
  1. Reproduces the original train/test split (same data cleaning, seed 42, stratified 80/20).
  2. Builds a *leak-free* test set: test rows whose cleaned text also occurs in the training
     set are removed (load_and_clean de-duplicates before cleaning, so some duplicates remain).
  3. Tunes Multinomial Naive Bayes (primary) and trains baselines on the SAME training rows:
     Complement NB, Logistic Regression, Linear SVM, and two dummy classifiers.
  4. Fits a Platt calibrator on out-of-fold NB predictions (training rows only).
  5. Writes metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier), confusion matrices,
     ROC / PR / reliability curves, threshold table, false-positive / false-negative analysis and a
     versioned record (data hash, params, seed, library versions) to reports/ and models/versions/.
Every number in the dashboard comes from these files; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from logger import get_logger
from mailguard.calibration import Calibrator, expected_calibration_error
from mailguard.explain import explain_prediction, raw_log_odds
from mailguard.registry import make_version_id, sha256_file, write_manifest
from preprocessing import load_and_clean

logger = get_logger(__name__)

SEED = 42
PRIMARY = "Multinomial Naive Bayes"


def tfidf():
    # identical to train.py so the primary model stays compatible with the shipped pipeline
    return TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))


def candidates():
    """name -> (estimator, param_grid or None)."""
    def pipe(clf):
        return Pipeline([("tfidf", tfidf()), ("clf", clf)])
    return {
        PRIMARY: (pipe(MultinomialNB()), {"clf__alpha": [0.1, 0.5, 1.0]}, 5),
        "Complement Naive Bayes": (pipe(ComplementNB()), {"clf__alpha": [0.1, 0.5, 1.0]}, 3),
        "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED)),
                                {"clf__C": [1.0, 10.0, 100.0]}, 3),
        "Linear SVM": (pipe(LinearSVC(random_state=SEED)), {"clf__C": [0.1, 1.0, 10.0]}, 3),
        "Baseline: stratified random": (pipe(DummyClassifier(strategy="stratified", random_state=SEED)), None, 0),
        "Baseline: always majority (ham)": (pipe(DummyClassifier(strategy="most_frequent")), None, 0),
    }


def scores_of(model, X):
    """Continuous spam score for ROC/PR, plus probability if the model has real probabilities."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba") and not isinstance(clf, LinearSVC):
        p = model.predict_proba(X)[:, 1]
        return p, p
    return model.decision_function(X), None


def bootstrap_f1_ci(y, pred, n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[idx], pred[idx], zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics_for(y, pred, score, prob):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, prob)) if prob is not None else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "f1_ci95": bootstrap_f1_ci(y, pred),
    }
    return out


def curve_points(y, score, kind, max_points=200):
    if kind == "roc":
        a, b, _ = roc_curve(y, score)
    else:
        b, a, _ = precision_recall_curve(y, score)  # b=precision, a=recall
        a, b = a, b
    a, b = np.round(a, 4), np.round(b, 4)
    step = max(1, len(a) // max_points)
    keep = list(range(0, len(a), step)) + [len(a) - 1]
    return {"x": a[keep].tolist(), "y": b[keep].tolist()}


def main(promote: bool = False, skip_cv_grid: bool = False) -> dict:
    t0 = time.time()
    df = load_and_clean(config.RAW_DATA_PATH)
    data_sha = sha256_file(config.RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["spam"], test_size=0.20, random_state=SEED, stratify=df["spam"])
    seen = set(X_train)
    leak_mask = X_test.isin(seen)
    X_eval, y_eval = X_test[~leak_mask], y_test[~leak_mask]
    logger.info("Rows=%d train=%d test=%d leak-free-test=%d (removed %d overlapping with train)",
                len(df), len(X_train), len(X_test), len(X_eval), int(leak_mask.sum()))

    results, curves, fitted, best_params = {}, {}, {}, {}
    for name, (est, grid, cv) in candidates().items():
        t1 = time.time()
        if grid:
            gs = GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=1)
            gs.fit(X_train, y_train)
            model, params, cv_f1 = gs.best_estimator_, gs.best_params_, float(gs.best_score_)
        else:
            model, params, cv_f1 = est.fit(X_train, y_train), {}, None
        fitted[name], best_params[name] = model, params
        s, p = scores_of(model, X_eval)
        pred = model.predict(X_eval)
        m = metrics_for(y_eval, pred, s, p)
        m.update({"best_params": params, "cv_f1": cv_f1, "train_seconds": round(time.time() - t1, 2),
                  "has_probabilities": p is not None})
        results[name] = m
        curves[name] = {"roc": curve_points(y_eval, s, "roc"), "pr": curve_points(y_eval, s, "pr")}
        logger.info("%s: F1=%.4f AUC=%.4f", name, m["f1"], m["roc_auc"] or float("nan"))

    primary = fitted[PRIMARY]

    # ---- parity with the shipped model + original-split (README) numbers --------------------
    shipped = joblib.load(config.MODEL_PATH)
    parity = {
        "agreement_with_retrained_on_test": float((shipped.predict(X_test) == primary.predict(X_test)).mean()),
        "shipped_alpha": float(shipped.named_steps["clf"].alpha),
        "retrained_alpha": float(primary.named_steps["clf"].alpha),
        "shipped_sklearn_note": "pickle created with scikit-learn 1.8.0 per requirements.txt",
    }
    sp = shipped.predict_proba(X_test)[:, 1]
    original_split = metrics_for(y_test, (sp >= .5).astype(int), sp, sp)
    original_split["n"] = int(len(y_test))

    # ---- calibration (fit on de-duplicated training rows, out-of-fold) ----------------------
    Xu = X_train[~X_train.duplicated()]
    yu = y_train.loc[Xu.index]
    oof = cross_val_predict(clone(primary), Xu, yu, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_log_proba")
    oof_lo = oof[:, 1] - oof[:, 0]
    version = make_version_id(data_sha, {"best": best_params, "tfidf": "stop_words=english,max_features=5000,ngram=(1,2)"},
                              SEED, sklearn.__version__)
    shipped_sha = sha256_file(config.MODEL_PATH)
    cal = Calibrator.fit(oof_lo, yu.values, meta={
        "for_model_sha": shipped_sha, "fitted_on": "out-of-fold NB log-odds, de-duplicated training rows",
        "n_fit": int(len(yu)), "version": version})

    lo_eval = raw_log_odds(shipped, X_eval)
    raw_p = shipped.predict_proba(X_eval)[:, 1]
    cal_p = cal.predict(lo_eval)
    ece_raw, rel_raw = expected_calibration_error(y_eval, raw_p)
    ece_cal, rel_cal = expected_calibration_error(y_eval, cal_p)
    calibration = {
        "method": "Platt scaling on NB log-odds", "a": cal.a, "b": cal.b,
        "brier_raw": float(brier_score_loss(y_eval, raw_p)), "brier_calibrated": float(brier_score_loss(y_eval, cal_p)),
        "ece_raw": ece_raw, "ece_calibrated": ece_cal,
        "reliability_raw": rel_raw, "reliability_calibrated": rel_cal,
        "label_agreement_raw_vs_calibrated": float(((raw_p >= .5) == (cal_p >= .5)).mean()),
    }

    # ---- primary model (shipped + calibrated) detailed analysis ---------------------------
    pred_c = (cal_p >= 0.5).astype(int)
    primary_cal = metrics_for(y_eval, pred_c, cal_p, cal_p)
    thresholds = []
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pr = (cal_p >= th).astype(int)
        cm = confusion_matrix(y_eval, pr, labels=[0, 1]).ravel()
        thresholds.append({"threshold": th, "precision": float(precision_score(y_eval, pr, zero_division=0)),
                           "recall": float(recall_score(y_eval, pr, zero_division=0)),
                           "f1": float(f1_score(y_eval, pr, zero_division=0)), "fp": int(cm[1]), "fn": int(cm[2])})

    errors = {"false_positives": [], "false_negatives": []}
    ev = pd.DataFrame({"text": X_eval.values, "y": y_eval.values, "p": cal_p})
    for kind, sel in (("false_positives", (ev.y == 0) & (ev.p >= .5)), ("false_negatives", (ev.y == 1) & (ev.p < .5))):
        for _, row in ev[sel].sort_values("p", ascending=(kind == "false_negatives")).head(15).iterrows():
            ex = explain_prediction(shipped, row.text, top_n=3)
            errors[kind].append({"snippet": row.text[:220], "length": len(row.text), "spam_probability": float(row.p),
                                 "spam_terms": [t for t, _ in ex.spam_terms], "ham_terms": [t for t, _ in ex.ham_terms]})
    errors["counts"] = {"false_positives": int(((ev.y == 0) & (ev.p >= .5)).sum()),
                        "false_negatives": int(((ev.y == 1) & (ev.p < .5)).sum())}

    nb = primary.named_steps["clf"]
    names = primary.named_steps["tfidf"].get_feature_names_out()
    delta = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top_features = {"spam": [(str(names[i]), float(delta[i])) for i in np.argsort(delta)[::-1][:15]],
                    "ham": [(str(names[i]), float(delta[i])) for i in np.argsort(delta)[:15]]}

    meta = {
        "version": version, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "split": "stratified 80/20, random_state=42 (same as train.py)",
        "dataset": {"file": config.RAW_DATA_PATH.name, "sha256": data_sha, "rows_after_cleaning": int(len(df)),
                    "train_rows": int(len(X_train)), "test_rows_original": int(len(X_test)),
                    "test_rows_leak_free": int(len(X_eval)), "removed_overlap": int(leak_mask.sum()),
                    "spam_rate": float(df["spam"].mean()), "eval_spam_rate": float(y_eval.mean())},
        "libs": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__},
        "train_seconds_total": round(time.time() - t0, 1),
    }
    evaluation = {"meta": meta, "primary": PRIMARY, "models": results, "curves": curves,
                  "primary_calibrated": primary_cal, "original_split_shipped_model": original_split,
                  "parity": parity, "calibration": calibration, "thresholds": thresholds,
                  "errors": errors, "top_features": top_features}

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    rows = [{"Model": k, **{c: v[c] for c in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
                                              "cv_f1", "train_seconds")}} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    # keep the original report format too
    pm = results[PRIMARY]
    (config.REPORTS_DIR / "tuning_results.json").write_text(json.dumps({
        "model": PRIMARY, "best_params": pm["best_params"], "best_cv_f1": pm["cv_f1"],
        "test_metrics_original_split_shipped_model": original_split,
        "test_metrics_leak_free": {k: pm[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}},
        indent=2), encoding="utf-8")

    vdir = config.VERSIONS_DIR / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(primary, vdir / "spam_classifier_pipeline.joblib")
    cal.save(vdir / "calibration.json")
    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (vdir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    if promote:
        shutil.copy2(vdir / "spam_classifier_pipeline.joblib", config.MODEL_PATH)
        cal.meta["for_model_sha"] = sha256_file(config.MODEL_PATH)
        logger.info("Promoted retrained model to %s", config.MODEL_PATH.name)
    cal.save(config.CALIBRATION_PATH)
    write_manifest(config.MODEL_PATH, config.MANIFEST_PATH,
                   {"version": version, "dataset_sha256": data_sha, "seed": SEED,
                    "primary_model": PRIMARY, "calibrated": True,
                    "parity_agreement_with_retrained": parity["agreement_with_retrained_on_test"]})
    logger.info("Done in %.1fs. Version %s", time.time() - t0, version)

    print(f"\nVersion: {version}\n")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nShipped model vs retrained agreement on test: {parity['agreement_with_retrained_on_test']:.4f}")
    print(f"Calibration  Brier {calibration['brier_raw']:.4f} -> {calibration['brier_calibrated']:.4f} | "
          f"ECE {ece_raw:.4f} -> {ece_cal:.4f}")
    return evaluation


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--promote", action="store_true", help="replace the shipped model with the retrained one")
    main(promote=ap.parse_args().promote)
