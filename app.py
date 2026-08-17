"""
app.py
------
Streamlit web application for the Spam Email Classification project.

Run with:
    streamlit run app.py
"""

import sys
import os
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import clean_text  # noqa: E402

MODEL_PATH = "models/spam_classifier_pipeline.joblib"

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main { padding-top: 1.5rem; }
    .result-spam {
        background-color: #FDECEC;
        border-left: 6px solid #DD3B3B;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-ham {
        background-color: #E9F7EF;
        border-left: 6px solid #2E9E5B;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .footer-note {
        color: #888;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    try:
        return pd.read_csv("reports/model_comparison.csv")
    except FileNotFoundError:
        return None


# ---------- Header ----------
st.title("📧 Spam Email Classification System")
st.write(
    "A Machine Learning system that classifies an email as **Spam** or **Ham "
    "(Not Spam)** based on its text content, built with a TF-IDF + Linear "
    "SVC pipeline trained on a real, publicly sourced email dataset."
)

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        - **Problem type:** Binary text classification (Spam vs Ham)
        - **Dataset:** ~5,700 real emails labeled spam/ham
        - **Pipeline:** TF-IDF vectorization → Linear Support Vector Classifier
        - **Tuning:** Hyperparameters optimized with GridSearchCV (5-fold CV, F1-scoring)
        - **Note:** Predictions are model estimates, not guaranteed to be 100% accurate.
        """
    )

st.divider()

# ---------- Input form ----------
st.subheader("Try it out")
email_text = st.text_area(
    "Paste an email's subject and/or body text below:",
    height=220,
    placeholder="Subject: ...\n\nDear customer, ...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)

if predict_clicked:
    if not email_text or not email_text.strip():
        st.error("Please enter some email text before predicting.")
    else:
        try:
            model = load_model()
            cleaned = clean_text(email_text)
            label = int(model.predict([cleaned])[0])

            score = None
            if hasattr(model, "decision_function"):
                score = float(model.decision_function([cleaned])[0])

            if label == 1:
                st.markdown(
                    f"""
                    <div class="result-spam">
                        <h4>🚫 Prediction: SPAM</h4>
                        <p>This email shows patterns consistent with spam content.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-ham">
                        <h4>✅ Prediction: HAM (Not Spam)</h4>
                        <p>This email looks like a legitimate message.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if score is not None:
                st.caption(
                    f"Model decision score: {score:.3f} "
                    "(further from 0 = more confident; negative leans Ham, positive leans Spam)"
                )

            st.info(
                "This is a Machine Learning estimate based on text patterns learned "
                "from the training data. It is not a guaranteed or legally binding "
                "spam determination."
            )
        except FileNotFoundError:
            st.error(
                "Trained model file not found. Please run `python src/train.py` "
                "first to generate models/spam_classifier_pipeline.joblib."
            )
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

st.divider()

# ---------- Model info ----------
st.subheader("Model Performance")
metrics_df = load_metrics()
if metrics_df is not None:
    st.dataframe(
        metrics_df.style.format(
            {c: "{:.4f}" for c in metrics_df.columns if c not in ("Model",)}
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("Model comparison results not found. Run the training pipeline to generate them.")

st.markdown(
    """
    <div class="footer-note">
    Built as an academic MCA-level Machine Learning project · Spam Email Classification System
    </div>
    """,
    unsafe_allow_html=True,
)
