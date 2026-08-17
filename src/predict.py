"""Command-line prediction using the trained Multinomial Naive Bayes pipeline."""

import sys
import joblib

from preprocessing import clean_text
from config import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_email(model, raw_text):
    text = clean_text(raw_text)
    label = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]

    return {
        "label": label,
        "label_name": "Spam" if label == 1 else "Ham (Not Spam)",
        "spam_probability": float(probabilities[1]),
        "ham_probability": float(probabilities[0]),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "email text here"')
        raise SystemExit(1)

    result = predict_email(load_model(), sys.argv[1])
    print(result)
