# Mail Guard AI

## Email Spam Classification System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://mail-guard-spam-filter-ai.streamlit.app/)
[![Repository](https://img.shields.io/badge/Source-GitHub-181717?logo=github&logoColor=white)](https://github.com/rati-ranjan-04/mail-guard-ai)

**Mail Guard AI** is an end-to-end Natural Language Processing and machine-learning project that classifies an email as **Spam** or **Ham (Not Spam)** from its text content. It includes data cleaning, exploratory analysis, TF-IDF feature extraction, model tuning, evaluation, model serialization, command-line inference, and a Streamlit web application.

> **Live application:** Try the deployed classifier at [mail-guard-spam-filter-ai.streamlit.app](https://mail-guard-spam-filter-ai.streamlit.app/).

The project is intended as an academic and portfolio demonstration of a reproducible text-classification workflow. The model should be treated as a decision-support tool rather than a replacement for a production email-security system.

## Project Overview

Unwanted and fraudulent email creates operational and security risks for individuals and organizations. This project addresses the problem as **binary text classification**:

| Label | Meaning |
| --- | --- |
| `0` | Ham — legitimate or non-spam email |
| `1` | Spam — unwanted or suspicious email |

The deployed application accepts pasted email text, applies the same preprocessing used during training, loads a serialized scikit-learn pipeline, and displays the predicted class together with Spam and Ham probabilities.

## Machine-Learning Workflow

The complete workflow is:

```text
Raw email CSV
    ↓
Load data and remove duplicate rows
    ↓
Normalize email text and remove empty records
    ↓
Stratified 80/20 train-test split
    ↓
TF-IDF word and bigram features
    ↓
Multinomial Naive Bayes classifier
    ↓
GridSearchCV tuning using five-fold cross-validation
    ↓
Evaluation on the held-out test set
    ↓
Save the fitted pipeline with joblib
    ↓
Streamlit application and command-line prediction
```

## Dataset

The repository contains the supplied dataset at `data/raw/emails.csv`. It has **5,728 data rows** and the columns `text` and `spam`. During training, duplicate rows are removed and the text is normalized; the training script also writes the cleaned dataset to `data/processed/emails_cleaned.csv`.

| Property | Value |
| --- | --- |
| Source file | `data/raw/emails.csv` |
| Data rows | 5,728 |
| Columns | `text`, `spam` |
| Target column | `spam` |
| Target values | `0 = Ham`, `1 = Spam` |
| Split | 80% training / 20% test |
| Random state | `42` |
| Split strategy | Stratified by target label |

The dataset is included for academic and reproducibility purposes. Before using the project with a different dataset, confirm that it provides the same `text` and `spam` columns and that labels are encoded as `0` and `1`.

## Text Preprocessing and Features

The shared function in `src/preprocessing.py` is used during both training and inference, which helps prevent differences between the model-training and serving paths. It lowercases each message, removes a leading `Subject:` token, collapses repeated whitespace, strips surrounding whitespace, removes duplicate rows, and discards empty messages.

The model pipeline uses `TfidfVectorizer` with English stop-word removal, a maximum vocabulary of 5,000 features, and word n-grams from one to two words. TF-IDF converts raw text into numerical features while giving greater weight to terms that are informative within the email corpus.

## Model and Hyperparameter Tuning

The final pipeline is implemented with scikit-learn:

```text
TfidfVectorizer
        ↓
MultinomialNB
```

`GridSearchCV` evaluates the Multinomial Naive Bayes smoothing parameter `clf__alpha` over `[0.1, 0.5, 1.0]`. Model selection uses the F1 score and five-fold cross-validation. The supplied training run selected `alpha = 0.1`.

The fitted pipeline is serialized to:

```text
models/spam_classifier_pipeline.joblib
```

Because the vectorizer and classifier are saved together, the application can receive raw email text without separately reproducing the feature-extraction steps.

## Evaluation Results

The following results are recorded in `reports/tuning_results.json` and were produced on the held-out test split from the included dataset.

| Metric | Score |
| --- | ---: |
| Accuracy | 98.95% |
| Precision | 98.16% |
| Recall | 97.45% |
| F1 score | 97.80% |
| ROC-AUC | 99.93% |
| Best cross-validation F1 | 96.97% |

The exact values are dataset- and split-specific. They should not be interpreted as guaranteed performance on future or real-world email traffic, particularly when email language, sender behavior, or spam tactics change.

Additional training artifacts are written to `reports/model_comparison.csv` and `reports/tuning_results.json`. Exploratory-analysis and model-visualization images are stored under `reports/figures/`.

## Streamlit Application

The user interface is defined in `app.py` and is available online at the [Mail Guard AI live demo](https://mail-guard-spam-filter-ai.streamlit.app/).

The application flow is:

1. Paste an email into the text area.
2. Select **Check Email**.
3. The saved pipeline cleans and classifies the message.
4. The interface displays **SPAM EMAIL** or **HAM — NOT SPAM**.
5. Spam and Ham probabilities are shown as percentages.

If the serialized model is missing, the application reports that the model must be trained first. Probabilities are model outputs and should not be treated as certainty.

## Repository Structure

```text
mail-guard-ai/
├── app.py                              # Streamlit user interface
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── .env                                # Local path and logging configuration
├── data/
│   ├── raw/
│   │   └── emails.csv                  # Input dataset
│   └── processed/
│       └── emails_cleaned.csv          # Generated cleaned dataset
├── models/
│   └── spam_classifier_pipeline.joblib # Serialized fitted pipeline
├── notebooks/
│   ├── EDA.ipynb                       # Exploratory data analysis
│   ├── Data_visualisation.ipynb        # Visual analysis
│   ├── data_cleaning.ipynb             # Data-cleaning workflow
│   └── spam_classification.ipynb       # Model-classification workflow
├── reports/
│   ├── figures/                         # Generated charts and evaluation plots
│   ├── model_comparison.csv            # Test metrics and training time
│   └── tuning_results.json             # Best parameters and evaluation results
├── logs/
│   └── app.log                         # Application and training logs
└── src/
    ├── config.py                        # Centralized paths and settings
    ├── logger.py                        # Logging configuration
    ├── predict.py                       # Command-line inference utility
    ├── preprocessing.py                 # Shared text-cleaning functions
    └── train.py                         # Training, tuning, evaluation, and saving
```

## Technologies

| Area | Technology |
| --- | --- |
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| Text features | scikit-learn TF-IDF |
| Classifier | Multinomial Naive Bayes |
| Model selection | GridSearchCV |
| Serialization | joblib |
| Visualization | Matplotlib, Seaborn, Jupyter Notebook |
| Application | Streamlit |
| Configuration | python-dotenv |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rati-ranjan-04/mail-guard-ai.git
=======
cd mail-guard-ai
>>>>>>> dbec0687e7f6d85d21af88bba3f805587fa24f6c
```

### 2. Create and activate a virtual environment

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository includes a `.env` file with default paths. If you create your own environment file, keep secrets and machine-specific configuration out of version control.

## Running the Project Locally

### Run the Streamlit application

The repository already contains a trained pipeline. Start the application from the project root:

```bash
python -m streamlit run app.py
```

Then open the local URL displayed by Streamlit, normally `http://localhost:8501`.

### Retrain the model

To rebuild the cleaned dataset, tune the model, evaluate it, save the reports, and serialize the pipeline, run:

```bash
python src/train.py
```

The command creates or updates:

```text
data/processed/emails_cleaned.csv
models/spam_classifier_pipeline.joblib
reports/model_comparison.csv
reports/tuning_results.json
logs/app.log
```

### Run command-line prediction

Pass the email text as a quoted command-line argument:

```bash
python src/predict.py "Subject: Congratulations! You have won a free prize!"
```

The utility returns a dictionary containing the numeric label, readable label, Spam probability, and Ham probability. It can also be imported from another Python program:

```python
from src.predict import load_model, predict_email

model = load_model()
result = predict_email(model, "Subject: Congratulations! You have won a free prize!")
print(result)
```

### Explore the notebooks

Install Jupyter if it is not already available, then run:

```bash
jupyter notebook
```

Open the notebooks in `notebooks/` to review the exploratory analysis, cleaning steps, visualizations, and classification workflow.

## Deployment

The deployed Streamlit application is connected to this project at [mail-guard-spam-filter-ai.streamlit.app](https://mail-guard-spam-filter-ai.streamlit.app/). A comparable deployment should include `app.py`, `requirements.txt`, the `src/` modules, and `models/spam_classifier_pipeline.joblib`.

When deploying a retrained model, verify that the deployment environment uses compatible versions of Python, scikit-learn, and joblib. The model artifact is generated with the dependency versions declared in `requirements.txt`.

## Limitations

This is an academic demonstration rather than a complete production email-security product. The classifier uses email text only; it does not inspect attachments, sender reputation, URLs, headers, authentication signals, or mailbox context. The included dataset is limited relative to modern email traffic, and spam tactics can change over time.

A high score on one held-out split does not guarantee equivalent performance in deployment. Before using a model in a high-impact workflow, evaluate it on representative, recent data, monitor false positives and false negatives, and establish an appropriate review process.

## Potential Improvements

Future work could add character-level features, richer word and character n-grams, additional classifiers such as linear SVM, threshold tuning, URL and domain features, email-header metadata, attachment signals, a larger and more recent dataset, monitoring, scheduled retraining, and authenticated API access.

## Academic Project Coverage

| Area | Implemented |
| --- | --- |
| Problem definition and business objective | Yes |
| Dataset inspection and cleaning | Yes |
| Duplicate and empty-text handling | Yes |
| Exploratory data analysis | Yes |
| Text preprocessing and TF-IDF features | Yes |
| Stratified train-test split | Yes |
| Multinomial Naive Bayes model | Yes |
| GridSearchCV hyperparameter tuning | Yes |
| Classification metrics and ROC-AUC | Yes |
| Model serialization with joblib | Yes |
| Streamlit application | Yes |
| Local command-line inference | Yes |
| Cloud deployment | Yes |

## References

[1]: https://github.com/rati-ranjan-04/mail-guard-ai "Mail Guard AI source repository"

[2]: https://mail-guard-spam-filter-ai.streamlit.app/ "Mail Guard AI Streamlit application"

[3]: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting "scikit-learn TF-IDF documentation"

[4]: https://scikit-learn.org/stable/modules/naive_bayes.html "scikit-learn Naive Bayes documentation"

[5]: https://docs.streamlit.io/ "Streamlit documentation"

## License

No license file is currently included in the repository. Add an explicit open-source license before redistributing the project or accepting external contributions.
