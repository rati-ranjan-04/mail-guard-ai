"""Probability calibration (Platt scaling on the NB log-odds), stored as plain JSON.

Multinomial Naive Bayes probabilities are usually over-confident. We fit a 1-D
logistic regression  p = sigmoid(a * log_odds + b)  on *out-of-fold* training
predictions, so the displayed confidence is a calibrated estimate rather than the
raw NB score. Stored as JSON (not pickle) so it carries no code-execution risk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CLIP = 60.0


def _sigmoid(z):
    z = np.clip(z, -CLIP, CLIP)
    return 1.0 / (1.0 + np.exp(-z))


class Calibrator:
    def __init__(self, a: float = 1.0, b: float = 0.0, meta: dict | None = None):
        self.a, self.b, self.meta = float(a), float(b), meta or {}

    def predict(self, log_odds):
        return _sigmoid(self.a * np.asarray(log_odds, dtype=float) + self.b)

    @classmethod
    def fit(cls, log_odds, y, meta: dict | None = None) -> "Calibrator":
        from sklearn.linear_model import LogisticRegression
        x = np.clip(np.asarray(log_odds, dtype=float), -CLIP, CLIP).reshape(-1, 1)
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
        lr.fit(x, np.asarray(y))
        return cls(float(lr.coef_[0][0]), float(lr.intercept_[0]), meta)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"method": "platt", "a": self.a, "b": self.b, "meta": self.meta}, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibrator | None":
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("method") != "platt" or not all(math.isfinite(float(data.get(k, "nan"))) for k in ("a", "b")):
            return None
        return cls(data["a"], data["b"], data.get("meta"))


def expected_calibration_error(y_true, p, bins: int = 10):
    """Return (ECE, reliability table [(mean_pred, frac_pos, count)])."""
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc, n = float(p[m].mean()), float(y_true[m].mean()), int(m.sum())
            ece += n / len(p) * abs(conf - acc)
            table.append((conf, acc, n))
    return float(ece), table
