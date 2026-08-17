# 📧 Spam Email Classification System Using Machine Learning

An end-to-end Machine Learning project (MCA-level academic project) that classifies emails as **Spam** or **Ham (not spam)** based on their text content, complete with EDA, a trained model pipeline, and a Streamlit web application.

> **Note on scope:** This project was originally scoped as a *salary prediction (regression)* project. The dataset actually available for this project (`emails.csv`) is a real spam/ham email dataset, not a salary dataset — so the project was built as a **spam classification (binary classification)** system instead, following the exact same end-to-end ML workflow and checklist. All numbers below are computed from the real dataset; nothing is fabricated.

---

## 1. Project Overview

Spam emails waste time, pose phishing/security risks, and clutter inboxes. This project builds a supervised ML system that automatically flags an email as spam or legitimate ("ham") based purely on its text, and exposes it through an interactive Streamlit demo.

## 2. Problem Statement

Given the raw text of an email, predict whether it is **spam (1)** or **ham (0)**.

**Business objective:** Email spam filtering protects users from phishing, scams, and unwanted marketing, and reduces the manual effort of sorting inboxes. A lightweight, interpretable classical-ML classifier (as opposed to a large model) is fast, cheap to run, and easy to explain/audit — a good fit for an email-filtering front line.

## 3. Objectives

- Build a clean, reproducible text-classification pipeline.
- Compare multiple classical ML algorithms fairly on the same data split.
- Tune the best model and quantify the improvement.
- Ship a usable Streamlit demo backed by the exact trained pipeline.

## 4. Dataset Information

- **File:** `emails.csv`
- **Size:** 5,728 rows × 2 columns (raw)
- **Columns:** `text` (raw email text, including a `Subject:` line) and `spam` (target: 1 = spam, 0 = ham)
- **Nature:** Real-world email text data (not synthetic)
- **Class balance:** 4,360 ham (76.1%) vs 1,368 spam (23.9%) — moderately imbalanced
- **Missing values:** none
- **Duplicate rows:** 33 (removed during cleaning)
- **After cleaning:** 5,695 rows (4,327 ham / 1,368 spam)

## 5. Features

- Paste any email text into the Streamlit app and get an instant Spam/Ham prediction.
- See the model's decision score and a plain-language interpretation.
- View live model comparison metrics inside the app.
- Robust error handling for empty/invalid input and a missing model file.

## 6. Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML | scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Random Forest, Linear SVC, GridSearchCV) |
| Model persistence | joblib |
| Web app | Streamlit |

## 7. Project Workflow

1. Data Collection → 2. Data Understanding → 3. EDA → 4. Data Cleaning → 5. Preprocessing (TF-IDF pipeline) → 6. Train/Test Split (80:20, stratified, `random_state=42`) → 7. Model Building (4 candidates) → 8. Model Evaluation → 9. Hyperparameter Tuning (GridSearchCV) → 10. Final Model Selection → 11. Save Pipeline → 12. Streamlit App → 13. Deployment.

## 8. EDA Summary

Key findings from `notebooks/EDA.ipynb` and `notebooks/Data_visualisation.ipynb` (see `reports/figures/` for all charts):

- The dataset is imbalanced (~76% ham / ~24% spam) — accounted for via **stratified splitting** and by evaluating with Precision/Recall/F1, not accuracy alone.
- Email length (character count and word count) differs on average between spam and ham, giving the model a useful signal alongside vocabulary.
- The most frequent non-stopword tokens differ meaningfully between spam and ham emails (see `05_top_words_spam.png` / `05_top_words_ham.png`), confirming vocabulary is a strong discriminative feature — which is exactly what TF-IDF captures.
- 33 exact duplicate emails were found and removed to avoid the same message appearing in both the train and test split.

## 9. Data Cleaning

| Issue | Action | Reason |
|---|---|---|
| 33 duplicate rows | Dropped | Prevents the same email leaking into both train and test sets |
| Mixed case text | Lowercased | Reduces vocabulary size / avoids treating "Free" and "free" as different tokens |
| Repeated `Subject:` prefix | Stripped | Constant across all rows, adds no signal, and would just become a boilerplate token |
| Irregular whitespace | Collapsed to single spaces | Cleaner tokenization |
| Any rows empty after cleaning | Dropped | An empty document carries no information for the model |

## 10. Data Preprocessing

- **Vectorization:** `TfidfVectorizer` (English stop words removed, uni+bi-grams, capped vocabulary) turns raw text into numeric features.
- **No leakage:** The vectorizer lives *inside* a scikit-learn `Pipeline`, so it is fit **only on the training fold** each time (including inside `GridSearchCV`'s cross-validation) — test data never influences the vocabulary or IDF weights.
- **Target:** `spam` column, used as-is (already binary).

## 11. Models Used & Comparison

Four candidate classifiers were trained on an identical 80:20 stratified split (`random_state=42`) inside the same TF-IDF pipeline:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Linear SVC** | 0.9921 | 0.9853 | 0.9818 | 0.9835 | 0.9997 |
| Random Forest | 0.9912 | 0.9818 | 0.9818 | 0.9818 | 0.9995 |
| Logistic Regression | 0.9860 | 0.9850 | 0.9562 | 0.9704 | 0.9994 |
| Multinomial Naive Bayes | 0.9851 | 0.9742 | 0.9635 | 0.9688 | 0.9988 |

*(Full table also saved at `reports/model_comparison.csv`; chart at `reports/figures/06_model_comparison.png`.)*

**Linear SVC** had the best F1 score and was carried forward for tuning.

## 12. Hyperparameter Tuning

`GridSearchCV` (5-fold CV, scored on F1) was run over the TF-IDF + Linear SVC pipeline:

- Grid: `tfidf__max_features ∈ {3000, 5000, 8000}`, `tfidf__ngram_range ∈ {(1,1), (1,2)}`, `clf__C ∈ {0.1, 0.5, 1, 5}`
- **Best params:** `max_features=8000`, `ngram_range=(1,1)`, `C=5`
- **Best CV F1:** 0.9839

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Accuracy | 0.9921 | **0.9956** |
| Precision | 0.9853 | **0.9963** |
| Recall | 0.9818 | 0.9854 |
| F1 Score | 0.9835 | **0.9908** |

**Tuning improved the model** — F1 rose from 0.9835 to 0.9908, driven mainly by a large precision gain (fewer false spam flags) with recall staying essentially level. Confusion matrix for the final model: `reports/figures/07_confusion_matrix.png`.

## 13. Final Model

**Final model: TF-IDF (max_features=8000, unigrams) + tuned Linear SVC (C=5)**

Selected because it had the best F1/Accuracy/Precision after tuning, trains quickly, and generalizes well on held-out data (5-fold CV score closely matches the test-set score, indicating low overfitting risk).

## 14. Model Saving

The complete pipeline (vectorizer + classifier) is saved with `joblib` to:

```
models/spam_classifier_pipeline.joblib
```

This single file is directly loadable and usable for inference — no retraining needed.

## 15. Installation Instructions

```bash
git clone <your-repo-url>
cd Spam-Classification
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 16. How to Train

```bash
python src/train.py
```

This regenerates `data/processed/emails_cleaned.csv`, retrains/compares all 4 models, tunes the best one, and overwrites `models/spam_classifier_pipeline.joblib`.

## 17. How to Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`). Paste in email text and click **Predict**.

## 18. Screenshots

_Add screenshots of the running Streamlit app here after your first local run, e.g._
`reports/figures/app_screenshot_1.png`

## 19. Deployment

To deploy on **Streamlit Community Cloud**:

1. Push this repository to GitHub (include `models/spam_classifier_pipeline.joblib` — it's a small file).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click **New app**.
3. Select this repository/branch and set the main file to `app.py`.
4. Streamlit Cloud automatically installs everything listed in `requirements.txt`.
5. Deploy — the app URL is shareable immediately.

(Any other host that runs `streamlit run app.py` — Render, Railway, a VM, etc. — works the same way; just ensure `requirements.txt` is installed first.)

## 20. Project Structure

```
Spam-Classification/
│
├── data/
│   ├── raw/
│   │   └── emails.csv
│   └── processed/
│       └── emails_cleaned.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── data_cleaning.ipynb
│   ├── Data_visualisation.ipynb
│   └── spam_classification.ipynb
│
├── models/
│   └── spam_classifier_pipeline.joblib
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   └── tuning_results.json
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 21. Future Improvements

- Add richer NLP features (character n-grams, email metadata like sender/links count) if such fields become available.
- Try a lightweight neural approach (e.g. fine-tuned small transformer) as an additional comparison point.
- Add explainability (e.g. top contributing tokens per prediction) to the Streamlit app.
- Set up CI to re-run `src/train.py` and flag metric regressions automatically.

## 22. Conclusion

This project demonstrates a complete, leakage-free ML workflow — from raw text to a tuned, saved model to a working web app — using only real, verifiable results from the provided dataset. The final tuned Linear SVC pipeline achieves **99.6% accuracy** and a **0.991 F1 score** on held-out test data, and is deployed through an interactive Streamlit interface for live demonstration.

## Final Academic Checklist

- ✅ Problem Statement
- ✅ Data Collection
- ✅ Data Understanding
- ✅ Exploratory Data Analysis
- ✅ Data Cleaning
- ✅ Data Preprocessing
- ✅ Train-Test Split
- ✅ Model Building
- ✅ Model Evaluation
- ✅ Hyperparameter Tuning
- ✅ Model Saving
- ✅ Streamlit Application
- ✅ Deployment Instructions
- ✅ GitHub-Ready Repository
