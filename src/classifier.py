"""Logistic-regression head over frozen (or SetFit-fine-tuned) embeddings.

`class_weight=balanced` matters here — the user's dataset has classes ranging
from 8 to 243 examples. Without it, the LR head will systematically under-predict
the tail.

A trained head bundles three things into one joblib artifact:
    - the sklearn LogisticRegression
    - the LabelMap (int↔str)
    - a small `meta` dict (encoder model name, dim, approach, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import LabelMap


@dataclass
class HeadBundle:
    model: LogisticRegression
    label_map: LabelMap
    meta: dict[str, Any]


def train_head(
    X: np.ndarray,
    y: np.ndarray,
    label_map: LabelMap,
    cfg: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> HeadBundle:
    """Fit an LR head on embedding matrix X and integer labels y."""
    tcfg = cfg["training"]
    clf = LogisticRegression(
        class_weight=tcfg.get("class_weight", "balanced"),
        max_iter=tcfg.get("max_iter", 2000),
        random_state=tcfg.get("seed", 42),
        n_jobs=-1,
    )
    clf.fit(X, y)
    return HeadBundle(model=clf, label_map=label_map, meta=dict(meta or {}))


def predict_proba(bundle: HeadBundle, X: np.ndarray) -> np.ndarray:
    """Return [N, C] class probabilities aligned with bundle.label_map ids."""
    proba = bundle.model.predict_proba(X)
    # sklearn orders columns by `bundle.model.classes_`; align to [0..C-1].
    classes = bundle.model.classes_
    if list(classes) != list(range(bundle.label_map.num_classes)):
        out = np.zeros((proba.shape[0], bundle.label_map.num_classes), dtype=proba.dtype)
        for col, cls in enumerate(classes):
            out[:, int(cls)] = proba[:, col]
        return out
    return proba


def save_head(bundle: HeadBundle, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "label_map": bundle.label_map.to_dict(),
            "meta": bundle.meta,
        },
        path,
    )
    # Also drop a human-readable label_map.json next to the bundle.
    (path.parent / "label_map.json").write_text(json.dumps(bundle.label_map.to_dict(), indent=2))
    return path


def load_head(path: Path | str) -> HeadBundle:
    blob = joblib.load(Path(path))
    return HeadBundle(
        model=blob["model"],
        label_map=LabelMap.from_dict(blob["label_map"]),
        meta=dict(blob.get("meta", {})),
    )
