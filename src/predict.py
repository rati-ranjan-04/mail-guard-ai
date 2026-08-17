"""Prediction utility for the MailGuard spam classifier."""

import sys
from pathlib import Path

import joblib

from config import MODEL_PATH
from preprocessing import clean_text


def load_model():
    """Load the trained spam classification pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please train the model first using:\n"
            "python src/train.py"
        )

    return joblib.load(MODEL_PATH)


def predict_email(model, email_text):
    """
    Predict whether an email is Spam or Ham.

    Returns:
        dict: Prediction label and probabilities.
    """

    if not email_text or not email_text.strip():
        raise ValueError("Email text cannot be empty.")

    cleaned_text = clean_text(email_text)

    prediction = int(
        model.predict([cleaned_text])[0]
    )

    probabilities = model.predict_proba(
        [cleaned_text]
    )[0]

    ham_probability = float(probabilities[0])
    spam_probability = float(probabilities[1])

    if prediction == 1:
        label = "Spam"
    else:
        label = "Ham"

    return {
        "label": prediction,
        "label_name": label,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
    }


def main():
    """Run prediction from the command line."""

    if len(sys.argv) < 2:
        print(
            'Usage: python src/predict.py "Your email text here"'
        )
        sys.exit(1)

    email_text = " ".join(sys.argv[1:])

    try:
        model = load_model()

        result = predict_email(
            model,
            email_text
        )

        print("\nPrediction Result")
        print("-----------------")
        print(
            f"Classification : {result['label_name']}"
        )
        print(
            f"Spam Probability : "
            f"{result['spam_probability'] * 100:.2f}%"
        )
        print(
            f"Ham Probability  : "
            f"{result['ham_probability'] * 100:.2f}%"
        )

    except FileNotFoundError as error:
        print(f"\nError: {error}")
        sys.exit(1)

    except ValueError as error:
        print(f"\nError: {error}")
        sys.exit(1)

    except Exception as error:
        print(
            f"\nPrediction failed: {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()