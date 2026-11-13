"""Exact, model-faithful explanation of a Multinomial Naive Bayes prediction.

For MultinomialNB the log-odds of spam vs ham decompose exactly into:
    log-odds = log P(spam)/P(ham)  +  sum_j  x_j * ( log P(w_j|spam) - log P(w_j|ham) )
where x_j is the TF-IDF value of term j. Each term's addend is its *contribution*.
Positive = pushes toward spam, negative = toward ham. No approximation is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Explanation:
    log_odds: float = 0.0
    prior_log_odds: float = 0.0
    n_known_terms: int = 0
    spam_terms: list = field(default_factory=list)   # [(term, contribution)]
    ham_terms: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"log_odds": self.log_odds, "prior_log_odds": self.prior_log_odds,
                "n_known_terms": self.n_known_terms, "spam_terms": self.spam_terms,
                "ham_terms": self.ham_terms, "note": self.note}


def _steps(pipeline):
    return pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]


def raw_log_odds(pipeline, texts) -> np.ndarray:
    """Spam-vs-ham log-odds for each cleaned text (numerically stable)."""
    _, clf = _steps(pipeline)
    logp = pipeline.predict_log_proba(list(texts))
    spam_idx = list(clf.classes_).index(1)
    ham_idx = list(clf.classes_).index(0)
    return logp[:, spam_idx] - logp[:, ham_idx]


def explain_prediction(pipeline, cleaned_text: str, top_n: int = 8) -> Explanation:
    tfidf, clf = _steps(pipeline)
    spam_idx = list(clf.classes_).index(1)
    ham_idx = list(clf.classes_).index(0)
    X = tfidf.transform([cleaned_text]).tocoo()
    delta = clf.feature_log_prob_[spam_idx] - clf.feature_log_prob_[ham_idx]
    prior = float(clf.class_log_prior_[spam_idx] - clf.class_log_prior_[ham_idx])

    exp = Explanation(prior_log_odds=prior, n_known_terms=int(X.nnz))
    if X.nnz == 0:
        exp.log_odds = prior
        exp.note = ("None of the message's words are in the model vocabulary, so the score "
                    "falls back to the class prior and is not informative.")
        return exp

    names = tfidf.get_feature_names_out()
    contrib = X.data * delta[X.col]
    exp.log_odds = prior + float(contrib.sum())
    order = np.argsort(contrib)
    pos = [(str(names[X.col[i]]), float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:top_n]
    neg = [(str(names[X.col[i]]), float(contrib[i])) for i in order if contrib[i] < 0][:top_n]
    exp.spam_terms, exp.ham_terms = pos, neg
    return exp
