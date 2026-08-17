"""
predict.py
----------
Load the saved pipeline and run predictions on new email text.
Used by app.py (Streamlit) and can also be run standalone:

    python src/predict.py "your email text here"
"""

import sys
import joblib
from preprocessing import clean_text

MODEL_PATH = "models/spam_classifier_pipeline.joblib"


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict_email(model, raw_text: str) -> dict:
    """
    Clean the input text with the SAME preprocessing used in training,
    then predict the class and (where available) a confidence score.
    """
    text = clean_text(raw_text)
    label = int(model.predict([text])[0])

    confidence = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        confidence = float(model.predict_proba([text])[0][label])
    elif hasattr(model, "decision_function"):
        # LinearSVC has no predict_proba; use decision_function distance as a proxy signal
        score = float(model.decision_function([text])[0])
        confidence = score

    return {
        "label": label,
        "label_name": "Spam" if label == 1 else "Ham (Not Spam)",
        "confidence_or_score": confidence,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"email text here\"")
        sys.exit(1)
    model = load_model()
    result = predict_email(model, sys.argv[1])
    print(result)
