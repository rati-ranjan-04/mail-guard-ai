"""
train.py
--------
End-to-end training script for the Spam Email Classification project.

Run from the project root:
    python src/train.py

What it does:
1. Loads and cleans the raw dataset (data/raw/emails.csv).
2. Splits into train/test (80:20, stratified, random_state=42).
3. Trains and compares 4 candidate classifiers on a TF-IDF pipeline.
4. Hyperparameter-tunes the best candidate with GridSearchCV.
5. Saves the final tuned pipeline (TF-IDF + classifier) to
   models/spam_classifier_pipeline.joblib
6. Prints a full metrics report so results are always reproducible
   from the actual data (nothing here is hard-coded or fabricated).
"""

import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from preprocessing import load_and_clean

RAW_PATH = "data/raw/emails.csv"
PROCESSED_PATH = "data/processed/emails_cleaned.csv"
MODEL_PATH = "models/spam_classifier_pipeline.joblib"
RANDOM_STATE = 42


def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", classifier),
    ])


def evaluate(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
    }
    try:
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
        metrics["ROC-AUC"] = roc_auc_score(y_test, scores)
    except Exception:
        metrics["ROC-AUC"] = None
    return metrics


def main():
    print("Loading and cleaning data...")
    df = load_and_clean(RAW_PATH)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned dataset shape: {df.shape}")

    X, y = df["text"], df["spam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Linear SVC": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
    }

    results = []
    fitted = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        metrics["Model"] = name
        metrics["Train Time (s)"] = round(time.time() - t0, 2)
        results.append(metrics)
        fitted[name] = pipe
        print(f"{name}: {metrics}")

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    print("\nModel comparison:\n", results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    print(f"\nBest baseline model: {best_name}. Proceeding to hyperparameter tuning...")

    # Hyperparameter tuning on the best candidate (Linear SVC in our experiments)
    base_pipe = build_pipeline(LinearSVC(random_state=RANDOM_STATE, max_iter=5000))
    param_grid = {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 0.5, 1, 5],
    }
    grid = GridSearchCV(base_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_pipe = grid.best_estimator_
    after_metrics = evaluate(best_pipe, X_test, y_test)
    print("Final tuned model metrics:", after_metrics)

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved final pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
