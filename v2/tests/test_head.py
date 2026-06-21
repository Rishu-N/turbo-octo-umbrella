"""Phase 5 tests: the learned head (src/classifier/head.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.classifier.head import load_head, predict_proba, save_head, train_head
from src.config import load_config


def _dataset(dim=16, per_class=20, seed=42):
    rng = np.random.default_rng(seed)
    labels = ["balance_inquiry", "transfer_funds", "card_lost_or_stolen"]
    bases = []
    X, y = [], []
    for ci, label in enumerate(labels):
        base = np.zeros(dim, dtype="float32")
        base[ci] = 1.0
        bases.append(base)
        for _ in range(per_class):
            v = base + 0.05 * rng.standard_normal(dim).astype("float32")
            X.append(v / np.linalg.norm(v))
            y.append(label)
    return np.array(X, dtype="float32"), y, bases


def _cfg(head_type="logreg", calibration="none"):
    cfg = load_config()
    cfg["combiner"]["head_type"] = head_type
    cfg["combiner"]["calibration"] = calibration
    return cfg


def test_logreg_predicts_clusters():
    X, y, bases = _dataset()
    bundle = train_head(X, y, _cfg("logreg"))
    p = predict_proba(bundle, bases[0].reshape(1, -1))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert bundle.labels[int(np.argmax(p[0]))] == "balance_inquiry"


def test_nearest_centroid_predicts_clusters():
    X, y, bases = _dataset()
    bundle = train_head(X, y, _cfg("nearest_centroid"))
    assert bundle.head_type == "nearest_centroid"
    p = predict_proba(bundle, bases[1].reshape(1, -1))
    assert bundle.labels[int(np.argmax(p[0]))] == "transfer_funds"


def test_no_calibration_keeps_temperature_one():
    X, y, _ = _dataset()
    bundle = train_head(X, y, _cfg("logreg", "none"))
    assert bundle.temperature == 1.0


def test_temperature_calibration_fits_and_keeps_valid_distribution():
    X, y, _ = _dataset()
    bundle = train_head(X[:45], y[:45], _cfg("logreg", "temperature"), X_val=X[45:], y_val=y[45:])
    assert bundle.temperature > 0 and np.isfinite(bundle.temperature)
    probs = predict_proba(bundle, X)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_save_load_roundtrip(tmp_path):
    X, y, _ = _dataset()
    bundle = train_head(X, y, _cfg("logreg"))
    path = str(tmp_path / "head.joblib")
    save_head(bundle, path)
    reloaded = load_head(path)
    assert reloaded.labels == bundle.labels and reloaded.head_type == bundle.head_type
    assert np.allclose(predict_proba(bundle, X[:5]), predict_proba(reloaded, X[:5]))
