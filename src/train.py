"""Train MailGuard using TF-IDF and Multinomial Naive Bayes."""

import sys
import time
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))


from preprocessing import load_and_clean
from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
)
from logger import get_logger


logger = get_logger(__name__)

RANDOM_STATE = 42


# --------------------------------------------------
# Build Model
# --------------------------------------------------

def build_pipeline():

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                MultinomialNB(),
            ),
        ]
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():

    print("=" * 55)
    print("MailGuard - Spam Email Classifier")
    print("Final Model: Multinomial Naive Bayes")
    print("=" * 55)

    logger.info(
        "Loading dataset from %s",
        RAW_DATA_PATH,
    )

    # Load and clean dataset
    df = load_and_clean(
        RAW_DATA_PATH
    )

    print(
        f"\nDataset size: {len(df)}"
    )

    # Save processed dataset
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    # Train-test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            df["text"],
            df["spam"],
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=df["spam"],
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # Build pipeline
    pipeline = build_pipeline()

    # Naive Bayes hyperparameter tuning
    param_grid = {
        "clf__alpha": [
            0.1,
            0.5,
            1.0,
        ]
    }

    print(
        "\nTraining Multinomial Naive Bayes..."
    )

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time() - start_time
    )

    # Best model
    best_model = (
        grid_search.best_estimator_
    )

    print(
        "\nBest Parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        f"\nBest CV F1 Score: "
        f"{grid_search.best_score_:.4f}"
    )

    # Evaluate
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    print(
        "\nTest Set Performance"
    )

    print("-" * 40)

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------
    # Save Reports
    # --------------------------------------------------

    reports_dir = (
        PROJECT_ROOT / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = {
        "Model": "Multinomial Naive Bayes",
        **metrics,
        "Best Parameters": str(
            grid_search.best_params_
        ),
        "CV F1 Score": grid_search.best_score_,
        "Training Time (seconds)": training_time,
    }

    pd.DataFrame(
        [report_data]
    ).to_csv(
        reports_dir
        / "model_comparison.csv",
        index=False,
    )

    tuning_results = {
        "model": "Multinomial Naive Bayes",
        "best_parameters": grid_search.best_params_,
        "best_cv_f1": grid_search.best_score_,
        "test_metrics": metrics,
    }

    with open(
        reports_dir
        / "tuning_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            tuning_results,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Final Verification
    # --------------------------------------------------

    classifier = (
        best_model.named_steps["clf"]
    )

    print(
        "\nFinal classifier:"
    )

    print(
        classifier
    )

    print("\nTraining completed successfully.")
    print("=" * 55)


if __name__ == "__main__":
    main()