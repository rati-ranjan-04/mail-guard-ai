# 📧 Spam Email Classification System

A simple academic Machine Learning project that classifies an email as **Spam** or **Ham (Not Spam)** using **Multinomial Naive Bayes**.

## 1. Problem Statement

Given the text of an email, predict whether it is spam or a legitimate email.

## 2. Dataset

- File: `data/raw/emails.csv`
- Rows: 5,728
- Columns: `text`, `spam`
- Target: `spam` (`1 = Spam`, `0 = Ham`)
- Real-world email text dataset
- Duplicate records are removed during preprocessing.

## 3. Machine Learning Workflow

```text
Email Text
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Multinomial Naive Bayes
   ↓
Spam / Ham Prediction
```

### Preprocessing

- Convert text to lowercase
- Remove the `Subject:` prefix
- Normalize whitespace
- Remove duplicate/empty records
- Convert text into numerical features using TF-IDF

### Model

**Multinomial Naive Bayes** is the final classification algorithm required for this project.

The model is tuned using `GridSearchCV` with 5-fold cross-validation. The main hyperparameter tuned is `alpha`.

Best parameter from the included training run:

```text
alpha = 0.1
```

## 4. Test Performance

The included trained model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 98.95% |
| Precision | 98.16% |
| Recall | 97.45% |
| F1 Score | 97.80% |
| ROC-AUC | 99.93% |

These values are generated from the supplied dataset and the saved Naive Bayes pipeline.

## 5. Streamlit Application

The app is intentionally kept simple:

1. Paste an email.
2. Click **Check Email**.
3. See **Spam** or **Ham**.
4. See Spam and Ham probabilities.

Run:

```bash
streamlit run app.py
```

## 6. Training

To retrain the model:

```bash
python src/train.py
```

The trained pipeline is saved to:

```text
models/spam_classifier_pipeline.joblib
```

## 7. Command-Line Prediction

```bash
python src/predict.py "Subject: You have won a free prize!"
```

## 8. Project Structure

```text
Spam-Classification/
│
├── data/
│   ├── raw/
│   │   └── emails.csv
│   └── processed/
│       └── emails_cleaned.csv
│
├── models/
│   └── spam_classifier_pipeline.joblib
│
├── reports/
│   ├── model_comparison.csv
│   └── tuning_results.json
│
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

## 9. Technologies

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- joblib
- Streamlit

## 10. Academic Summary

**Final Algorithm:** Multinomial Naive Bayes

**Feature Extraction:** TF-IDF

**Problem Type:** Binary Text Classification

**Classes:** Spam / Ham

**Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC
