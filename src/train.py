"""Train a simple Spam Email Classifier using Multinomial Naive Bayes."""

import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import load_and_clean
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH
from logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


def main():
    logger.info("Loading data from %s", RAW_DATA_PATH)
    df = load_and_clean(RAW_DATA_PATH)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["spam"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["spam"],
    )

    model = build_pipeline()
    param_grid = {
        "clf__alpha": [0.1, 0.5, 1.0],
    }

    logger.info("Tuning Multinomial Naive Bayes...")
    start = time.time()
    grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    metrics = evaluate(best_model, X_test, y_test)
    metrics["Model"] = "Multinomial Naive Bayes"
    metrics["Train Time (s)"] = round(time.time() - start, 2)

    print("\nFinal Model: Multinomial Naive Bayes")
    print("Best Parameters:", grid.best_params_)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    reports = MODEL_PATH.parent.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(reports / "model_comparison.csv", index=False)

    import json
    with open(reports / "tuning_results.json", "w", encoding="utf-8") as file:
        json.dump({
            "model": "Multinomial Naive Bayes",
            "best_params": grid.best_params_,
            "best_cv_f1": grid.best_score_,
            "test_metrics": metrics,
        }, file, indent=2)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Saved Naive Bayes model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
