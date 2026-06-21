"""Learned head over Jina embeddings — logistic regression / nearest-centroid + calibration.  [Phase 5]

Alongside the pure NN cache, a lightweight head generalizes on confusable intents. Both train in seconds
on cached embeddings, and a new intent is added by refitting the head or adding a centroid — NEVER by
training a base model (§13). Default is logistic regression (config: combiner.head_type).

Confidences are temperature-calibrated (config: combiner.calibration) so the head's probabilities are
comparable to the thresholds: `predict_proba = softmax(logits / T)`. `T` is fit on a held-out set by
minimizing NLL; `T = 1` (default) reproduces the uncalibrated distribution.

Public surface:
    train_head(X, y, cfg, X_val=None, y_val=None) -> HeadBundle
    predict_proba(bundle, X) -> np.ndarray     # [N, C] calibrated probabilities, columns = bundle.labels
    calibrate_temperature(bundle, X_val, y_val) -> float
    save_head(bundle, path) / load_head(path) -> HeadBundle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np


@dataclass
class HeadBundle:
    """The trained head plus the label ordering, head type, and calibration needed to interpret it."""

    model: Any                 # sklearn LogisticRegression | centroids ndarray [C, dim] (nearest_centroid)
    labels: list[str]          # class order: column j of predict_proba corresponds to labels[j]
    head_type: str             # "logreg" | "nearest_centroid"
    temperature: float = 1.0   # softmax temperature (1.0 = uncalibrated)
    meta: dict[str, Any] = field(default_factory=dict)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _logits(bundle: HeadBundle, X: np.ndarray) -> np.ndarray:
    """Pre-temperature logits [N, C] aligned with ``bundle.labels``."""
    X = np.asarray(X, dtype=np.float32)
    if bundle.head_type == "nearest_centroid":
        return X @ bundle.model.T  # cosine similarities (X and centroids are L2-normalized)
    base = bundle.model.predict_proba(X)  # columns already in bundle.labels order (== clf.classes_)
    return np.log(np.clip(base, 1e-9, 1.0))


def predict_proba(bundle: HeadBundle, X: np.ndarray) -> np.ndarray:
    """Return [N, C] temperature-calibrated class probabilities aligned with ``bundle.labels``."""
    return _softmax(_logits(bundle, X) / max(bundle.temperature, 1e-6), axis=1)


def calibrate_temperature(bundle: HeadBundle, X_val: np.ndarray, y_val: Any) -> float:
    """Fit the softmax temperature on a held-out set by minimizing NLL; updates and returns it."""
    from scipy.optimize import minimize_scalar

    z = _logits(bundle, np.asarray(X_val, dtype=np.float32))
    idx = {label: i for i, label in enumerate(bundle.labels)}
    y_idx = np.array([idx[str(t)] for t in y_val])

    def nll(temp: float) -> float:
        p = _softmax(z / max(temp, 1e-6), axis=1)
        return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    bundle.temperature = float(res.x)
    return bundle.temperature


def train_head(
    X: np.ndarray, y: Any, cfg: dict[str, Any], X_val: np.ndarray | None = None, y_val: Any | None = None
) -> HeadBundle:
    """Fit the configured head (combiner.head_type) on embedding matrix X and labels y."""
    head_type = cfg["combiner"]["head_type"]
    seed = int(cfg.get("seed", 42))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray([str(t) for t in y])

    if head_type == "nearest_centroid":
        labels = sorted(set(y.tolist()))
        centroids = np.zeros((len(labels), X.shape[1]), dtype=np.float32)
        for i, label in enumerate(labels):
            c = X[y == label].mean(axis=0)
            norm = np.linalg.norm(c)
            centroids[i] = c / norm if norm > 0 else c
        bundle = HeadBundle(model=centroids, labels=labels, head_type="nearest_centroid",
                            meta={"n": int(len(y))})
    else:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        clf.fit(X, y)
        bundle = HeadBundle(model=clf, labels=[str(c) for c in clf.classes_], head_type="logreg",
                            meta={"n": int(len(y))})

    if (
        cfg["combiner"].get("calibration", "none") == "temperature"
        and X_val is not None
        and y_val is not None
        and len(X_val) > 0
    ):
        calibrate_temperature(bundle, X_val, y_val)
    return bundle


def save_head(bundle: HeadBundle, path: str) -> None:
    """Persist the head bundle (joblib) to ``path`` (config: combiner.head_path)."""
    joblib.dump(bundle, path)


def load_head(path: str) -> HeadBundle:
    """Load a head bundle previously saved with ``save_head``."""
    return joblib.load(path)
