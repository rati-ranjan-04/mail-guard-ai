"""Simple Streamlit interface for the Spam Email Classifier."""

import os
import sys
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import clean_text
from config import MODEL_PATH

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
)

st.title("📧 Spam Email Classifier")
st.caption("TF-IDF + Multinomial Naive Bayes")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

email = st.text_area(
    "Paste email text",
    height=220,
    placeholder="Subject: Congratulations! You have won a prize...",
)

if st.button("Check Email", type="primary", use_container_width=True):
    if not email.strip():
        st.warning("Please enter an email first.")
    else:
        try:
            model = load_model()
            text = clean_text(email)
            prediction = int(model.predict([text])[0])
            probabilities = model.predict_proba([text])[0]

            spam_probability = probabilities[1] * 100
            ham_probability = probabilities[0] * 100

            st.divider()

            if prediction == 1:
                st.error("🚨 SPAM EMAIL")
            else:
                st.success("✅ HAM — NOT SPAM")

            col1, col2 = st.columns(2)
            col1.metric("Spam Probability", f"{spam_probability:.2f}%")
            col2.metric("Ham Probability", f"{ham_probability:.2f}%")

        except FileNotFoundError:
            st.error("Model not found. Run `python src/train.py` first.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()
st.caption("Academic Machine Learning Project • Multinomial Naive Bayes")
