"""Polished Streamlit interface for the Spam Email Classifier."""

import os
import sys

import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import clean_text
from config import MODEL_PATH


st.set_page_config(
    page_title="SpamGuard | Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Styling ----------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --primary: #6d5dfc;
            --primary-dark: #5145cd;
            --text: #172033;
            --muted: #687386;
            --surface: #ffffff;
            --border: #e6e9f0;
            --background: #f6f7fb;
        }

        .stApp {
            background: var(--background);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding: 2.5rem 2rem 2rem;
        }

        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #181b3a 0%, #373083 52%, #6d5dfc 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(45, 39, 122, 0.22);
            margin-bottom: 1.5rem;
        }

        .hero-kicker {
            color: #c9c5ff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.08;
            margin: 0 0 0.75rem;
            font-weight: 800;
        }

        .hero p {
            color: #e7e6ff;
            font-size: 1.05rem;
            margin: 0;
            max-width: 680px;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text);
            margin: 0.4rem 0 0.7rem;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 8px 25px rgba(25, 31, 51, 0.05);
            height: 100%;
        }

        .card-icon {
            font-size: 1.45rem;
            margin-bottom: 0.35rem;
        }

        .card-title {
            color: var(--text);
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .card-text {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .result-card {
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-top: 1.25rem;
            border: 1px solid var(--border);
            background: #fff;
        }

        .result-spam {
            border-left: 6px solid #ef4444;
            background: linear-gradient(90deg, #fff5f5, #ffffff);
        }

        .result-ham {
            border-left: 6px solid #16a34a;
            background: linear-gradient(90deg, #f1fff6, #ffffff);
        }

        .result-label {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--text);
        }

        .result-description {
            color: var(--muted);
            margin-top: 0.3rem;
        }

        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            color: white;
            font-weight: 700;
            min-height: 3rem;
            box-shadow: 0 8px 18px rgba(109, 93, 252, 0.25);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-dark), #7c3aed);
            transform: translateY(-1px);
        }

        /* High-contrast email input */
        [data-testid="stTextArea"] {
            margin-bottom: 0.75rem;
        }

        [data-testid="stTextArea"] label {
            color: var(--text) !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
        }

        [data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            caret-color: #6d5dfc !important;
            border: 2px solid #cfd5e2 !important;
            border-radius: 14px !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            padding: 1rem !important;
            box-shadow: 0 5px 16px rgba(25, 31, 51, 0.06) !important;
        }

        [data-testid="stTextArea"] textarea::placeholder {
            color: #7a8496 !important;
            -webkit-text-fill-color: #7a8496 !important;
            opacity: 1 !important;
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: #6d5dfc !important;
            box-shadow: 0 0 0 3px rgba(109, 93, 252, 0.18) !important;
            outline: none !important;
        }

        [data-testid="stMetric"] {
            background: #f8f9fd;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.75rem 1rem;
        }

        .footer {
            text-align: center;
            color: #8a93a5;
            font-size: 0.78rem;
            padding: 1.8rem 0 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📧 SpamGuard")
    st.caption("A machine-learning powered email screening tool.")
    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. Paste the email content.\n"
        "2. Click **Analyze Email**.\n"
        "3. Review the predicted class and confidence scores."
    )

    st.divider()
    st.markdown("### Model information")
    st.info("**Algorithm:** Multinomial Naive Bayes\n\nThe classifier analyzes cleaned text and returns spam and ham probabilities.")

    if st.button("Clear email"):
        st.session_state.email_text = ""
        st.rerun()


# ---------- Main content ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Email security assistant</div>
        <h1>Detect suspicious emails<br>before they reach your inbox.</h1>
        <p>Paste an email below and let the classifier assess whether it looks like spam or a legitimate message.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Analyze an email</div>', unsafe_allow_html=True)
    st.caption("Paste the subject and full message below. Your input will appear in a high-contrast editor.")
    email = st.text_area(
        "Email content",
        key="email_text",
        height=255,
        label_visibility="visible",
        placeholder="Subject: Congratulations! You have won a prize...",
    )

    analyze = st.button("🔍  Analyze Email", type="primary")

with right:
    st.markdown('<div class="section-title">Quick guidance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="card-icon">🧠</div><div class="card-title">Use the full message</div><div class="card-text">Including the subject line and body usually gives the model more context.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="card-icon">⚠️</div><div class="card-title">Treat results as guidance</div><div class="card-text">Do not open links or attachments in suspicious messages, even when a message is classified as ham.</div></div>',
        unsafe_allow_html=True,
    )


if analyze:
    if not email.strip():
        st.warning("Please enter an email before analyzing it.")
    else:
        try:
            with st.spinner("Analyzing email content..."):
                model = load_model()
                text = clean_text(email)
                prediction = int(model.predict([text])[0])
                probabilities = model.predict_proba([text])[0]

            spam_probability = float(probabilities[1] * 100)
            ham_probability = float(probabilities[0] * 100)

            if prediction == 1:
                st.markdown(
                    '<div class="result-card result-spam"><div class="result-label">🚨 Likely spam email</div><div class="result-description">This message contains patterns commonly associated with spam.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-ham"><div class="result-label">✅ Likely legitimate email</div><div class="result-description">This message is more consistent with the ham/not-spam examples seen by the model.</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### Classification confidence")
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Spam probability", f"{spam_probability:.2f}%")
            metric_right.metric("Ham probability", f"{ham_probability:.2f}%")

            st.progress(spam_probability / 100, text=f"Spam likelihood: {spam_probability:.2f}%")

        except FileNotFoundError:
            st.error("Model not found. Run `python src/train.py` first.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


st.markdown(
    '<div class="footer">Academic Machine Learning Project · Multinomial Naive Bayes · Rati Ranjan Mohapatra · IPSAR · MCA</div>',
    unsafe_allow_html=True,
)
