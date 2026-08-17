📧 Email Spam Classification System

Academic Machine Learning Project — Binary Text Classification

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies an email as Spam or Ham (Not Spam) from its text content.

The project covers the complete ML lifecycle: data loading → cleaning → EDA → TF-IDF feature extraction → model training → hyperparameter tuning → evaluation → model serialization → Streamlit deployment.

1. Project Overview

Problem Statement

Unwanted and fraudulent emails are a common problem for individuals and organizations. Manually identifying spam is inefficient and unreliable at scale.

This project builds a supervised Machine Learning classifier that reads email text and predicts whether the message is:

Spam (1): unwanted/suspicious email

Ham (0): legitimate email

Business Objective

The objective is to develop a lightweight automated spam-filtering system that can assist users in identifying potentially unwanted emails before they are opened or processed further.

Problem Type

Binary Text Classification

2. Dataset

The project uses the supplied emails.csv dataset.

Property

Details

File

data/emails.csv

Initial rows

5,728

Columns

text, spam

Target

spam

Target values

0 = Ham, 1 = Spam

Data type

Real-world email text data

Duplicate handling

Duplicate rows removed during training

After duplicate removal, 5,695 records were used for modeling.

Note: The dataset is included in this project for academic/reproducibility purposes. If the dataset is too large or distribution is restricted for a GitHub submission, upload it separately or provide its source link instead.

3. Machine Learning Workflow

Raw Email Dataset
       │
       ▼
Data Understanding
       │
       ▼
EDA & Data Quality Checks
       │
       ▼
Remove Duplicates / Handle Missing Text
       │
       ▼
Train-Test Split (80:20)
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Logistic Regression
       │
       ▼
GridSearchCV Hyperparameter Tuning
       │
       ▼
Model Evaluation
       │
       ▼
Save Pipeline with Joblib
       │
       ▼
Streamlit Web Application
       │
       ▼
Prediction: Spam / Ham

4. Data Understanding & EDA

The project notebook performs the required exploratory analysis, including:

Dataset shape and column inspection

Data types

Target-variable distribution

Missing-value analysis

Duplicate analysis

Email text length / word-count analysis

Univariate analysis

Class distribution visualization

Basic text-based feature analysis

Confusion-matrix visualization after model evaluation

Generated Reports

The reports/ directory contains generated artifacts such as:

reports/
├── avg_word_count.png
├── class_distribution.png
├── confusion_matrix.png
└── metrics.json

Each visualization is intended to answer a specific analytical question rather than being included only for presentation.

5. Data Cleaning & Preprocessing

The training pipeline performs the following data-quality steps:

Load the CSV file using pandas.

Remove duplicate rows.

Replace missing email text with an empty string.

Convert email text to string format.

Convert the target variable to integer format.

Split the data using stratification to preserve the Spam/Ham class ratio.

Convert text into numerical features using TF-IDF.

Why TF-IDF?

Machine Learning algorithms cannot directly work with raw email text. TF-IDF converts text into numerical features based on the importance of words within the dataset.

The vectorizer also considers the selected n-gram configuration during hyperparameter tuning.

6. Model Building

Final Algorithm

Logistic Regression was selected as the final classifier.

It is well suited for binary classification and performs effectively on high-dimensional sparse text features such as TF-IDF vectors.

Pipeline

The model is implemented as a scikit-learn Pipeline:

TfidfVectorizer
      ↓
LogisticRegression

Keeping preprocessing and classification inside one pipeline helps ensure that the same transformations are automatically applied during training and prediction.

7. Train-Test Split

The cleaned dataset is divided into:

80% Training data: 4,556 records

20% Testing data: 1,139 records

The split uses stratify=y so that the Spam/Ham class distribution remains approximately consistent in both sets.

train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

8. Hyperparameter Tuning

GridSearchCV is used to select a better-performing configuration.

The search explores:

TF-IDF n-gram range:
    (1,1)
    (1,2)

Logistic Regression C:
    1.0
    2.0

The tuning objective is F1 Score, which provides a balance between precision and recall.

Best Configuration

Logistic Regression C = 2.0
TF-IDF n-gram range = (1, 1)

Cross-validation is performed using 3 folds in the supplied training script.

9. Model Evaluation

The tuned model is evaluated on the unseen test set using the required classification metrics.

Metric

Score

Accuracy

99.39%

Precision

99.63%

Recall

97.81%

F1 Score

98.71%

ROC-AUC

99.99%

Confusion Matrix

                 Predicted
                 Ham   Spam
Actual Ham       864     1
Actual Spam        6   268

The model correctly identifies most legitimate and spam emails while producing very few false classifications on the supplied test split.

These metrics describe the included training run and should not be interpreted as guaranteed performance on unseen real-world email traffic.

10. Model Saving

The complete tuned preprocessing + classification pipeline is saved using joblib:

models/spam_classifier.joblib

Because the TF-IDF vectorizer and classifier are stored together, the application can directly load the saved pipeline and perform predictions on raw email text.

11. Streamlit Application

The project includes a simple web interface built with Streamlit.

Application Flow

User pastes email
       ↓
Click Predict
       ↓
Saved ML Pipeline
       ↓
TF-IDF Transformation
       ↓
Logistic Regression
       ↓
Spam / Ham + Spam Probability

The interface displays:

Predicted class

Spam probability

Basic input validation

A warning that ML predictions should not be treated as absolute guarantees

12. Project Structure

email_spam_classifier_project/
│
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
│
├── data/
│   └── emails.csv                 # Dataset
│
├── models/
│   └── spam_classifier.joblib     # Trained ML pipeline
│
├── notebooks/
│   └── email_spam_detection.ipynb # EDA + ML workflow
│
├── reports/
│   ├── avg_word_count.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── metrics.json
│
└── src/
    ├── train_model.py             # Model training + tuning
    └── predict.py                  # Prediction utility

13. Technologies Used

Python 3.10+

pandas — data loading and manipulation

NumPy — numerical operations

scikit-learn — preprocessing, model training, tuning and evaluation

TF-IDF — text feature extraction

Logistic Regression — binary classifier

GridSearchCV — hyperparameter tuning

joblib — model serialization

Matplotlib — visualization

Jupyter Notebook — experimentation and EDA

Streamlit — web application

14. Installation & Setup

Step 1 — Clone the repository

git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier

Replace <your-username> and repository name with your actual GitHub details.

Step 2 — Create a virtual environment

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Windows CMD

python -m venv .venv
.venv\Scripts\activate

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Step 3 — Upgrade pip

python -m pip install --upgrade pip

Step 4 — Install dependencies

pip install -r requirements.txt

15. Run the Project

Option A — Run the Streamlit Application

If the saved model already exists:

streamlit run app.py

Then open the local URL displayed by Streamlit, normally:

http://localhost:8501

Option B — Retrain the Model

To train the model again from the dataset:

python .\src\train.py

The script will:

Load the dataset

Remove duplicates

Perform the train-test split

Build the TF-IDF + Logistic Regression pipeline

Run GridSearchCV

Evaluate the model

Save the trained pipeline

Save evaluation metrics

Outputs:

models/spam_classifier.joblib
reports/metrics.json

Option C — Run Command-Line Prediction

python src/predict.py

For a custom prediction, the prediction helper can also be imported into another Python program:

from src.predict import predict_email

label, probability = predict_email(
    "Subject: Congratulations! You have won a free prize!"
)

print("Prediction:", label)
print("Spam probability:", probability)

16. Run the Jupyter Notebook

Start Jupyter:

jupyter notebook

Then open:

notebooks/email_spam_detection.ipynb

The notebook can be used to demonstrate the academic workflow from data understanding through model evaluation.

17. GitHub Setup

After completing the project locally:

git init
git add .
git commit -m "Add email spam classification ML project"
git branch -M main
git remote add origin https://github.com/<your-username>/email-spam-classifier.git
git push -u origin main

Before pushing, verify the repository contents:

git status
git ls-files

Recommended .gitignore

.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
.env

If your dataset is too large for GitHub, do not force-add it. Instead, provide the dataset source/download instructions in this README.

18. Deployment

The Streamlit app can be deployed to a cloud platform that supports Python/Streamlit applications.

Typical deployment flow:

GitHub Repository
       ↓
Cloud Deployment Platform
       ↓
Install requirements.txt
       ↓
Run app.py
       ↓
Public Streamlit Application URL

For deployment, make sure these files are committed:

app.py
requirements.txt
models/spam_classifier.joblib
python .\src\predict.py

If the deployment platform requires a specific Python version, pin the version according to that platform's current supported runtime.

19. Limitations

This project is designed primarily as an academic demonstration.

The dataset is limited compared with the volume and diversity of modern email traffic.

Spam patterns can change over time.

Text-only classification cannot inspect attachments, sender reputation, URLs, headers, or other email metadata.

Model probabilities should not be treated as absolute certainty.

Very high test performance on one dataset does not guarantee equivalent production performance.

20. Possible Future Improvements

The system could be extended with:

Character-level TF-IDF features

Word + character n-gram combination

Naive Bayes and Linear SVM model comparison

Class-weight tuning

Threshold optimization for business requirements

URL and domain features

Email-header metadata

Attachment-related features

Larger and more recent datasets

Model monitoring and periodic retraining

Docker-based deployment

Authentication and API integration

21. Academic Checklist Coverage

Requirement

Status

Problem Statement

✅

Business Objective

✅

Dataset Collection

✅

Data Understanding

✅

EDA

✅

Missing Values

✅

Duplicate Handling

✅

Data Cleaning

✅

Text Preprocessing

✅

Feature Extraction

✅ TF-IDF

Train-Test Split

✅

Model Building

✅ Logistic Regression

Classification Metrics

✅

Confusion Matrix

✅

Hyperparameter Tuning

✅ GridSearchCV

Model Saving

✅ joblib

Streamlit Application

✅

Deployment Ready

✅

GitHub Structure

✅

22. Final Project Summary

Project: Email Spam Classification System
Domain: Natural Language Processing / Machine Learning
Problem: Binary text classification
Input: Email text
Output: Spam or Ham
Feature Extraction: TF-IDF
Final Model: Logistic Regression
Hyperparameter Tuning: GridSearchCV
Model Storage: Joblib
Application: Streamlit

The project demonstrates a complete, reproducible machine-learning workflow suitable for an academic submission and as a foundation for a more advanced spam-filtering application.