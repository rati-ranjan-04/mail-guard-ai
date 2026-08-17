"""Streamlit application for MailGuard Spam Email Classifier."""

import sys
from pathlib import Path

import joblib
import streamlit as st


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


from config import MODEL_PATH
from preprocessing import clean_text


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MailGuard Spam Classifier",
    page_icon="📧",
    layout="centered",
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained Multinomial Naive Bayes model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Page Header
# --------------------------------------------------

st.title("📧 MailGuard")

st.subheader("Spam Email Classifier")

st.caption(
    "TF-IDF + Multinomial Naive Bayes"
)

st.write(
    "Paste an email below to check whether it is "
    "Spam or Ham."
)


# --------------------------------------------------
# Email Input
# --------------------------------------------------

email_text = st.text_area(
    "Email Content",
    height=250,
    placeholder=(
        "Paste the complete email here...\n\n"
        "Example:\n"
        "Congratulations! You have won a prize. "
        "Click the link below to claim your reward."
    ),
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Check Email",
    type="primary",
    use_container_width=True,
):

    if not email_text.strip():

        st.warning(
            "Please enter an email before checking."
        )

    else:

        try:

            # Load trained model
            model = load_model()

            # Clean email text
            cleaned_text = clean_text(
                email_text
            )

            # Prediction
            prediction = int(
                model.predict(
                    [cleaned_text]
                )[0]
            )

            # Probability prediction
            probabilities = model.predict_proba(
                [cleaned_text]
            )[0]

            ham_probability = (
                probabilities[0] * 100
            )

            spam_probability = (
                probabilities[1] * 100
            )

            # --------------------------------------
            # Result
            # --------------------------------------

            st.divider()

            if prediction == 1:

                st.error(
                    "🚨 SPAM EMAIL"
                )

                st.write(
                    "This email is likely to be spam."
                )

            else:

                st.success(
                    "✅ HAM — NOT SPAM"
                )

                st.write(
                    "This email appears to be legitimate."
                )

            # --------------------------------------
            # Probability Metrics
            # --------------------------------------

            st.subheader(
                "Prediction Confidence"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2f}%",
                )

            with col2:

                st.metric(
                    "Ham Probability",
                    f"{ham_probability:.2f}%",
                )

            # --------------------------------------
            # Probability Bar
            # --------------------------------------

            st.progress(
                int(round(spam_probability))
            )

            st.caption(
                "Spam probability"
            )

        except FileNotFoundError:

            st.error(
                "❌ Trained model not found."
            )

            st.info(
                "Run the training script first:"
            )

            st.code(
                "python src/train.py",
                language="powershell",
            )

        except Exception as error:

            st.error(
                "❌ Prediction failed."
            )

            st.caption(
                f"Error: {error}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "MailGuard • Academic Machine Learning Project"
)

st.caption(
    "Model: Multinomial Naive Bayes | "
    "Features: TF-IDF"
)