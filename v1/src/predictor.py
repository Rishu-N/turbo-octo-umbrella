"""Stage-1 predictor: load the trained encoder + LR head and score intents.

Shared by `evaluate.py` and `reranker.py` so the "embed → LR head → probabilities"
logic lives in exactly one place. This is the embedding-classifier stage; the
reranker (when enabled) is a second stage layered on top of these scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .classifier import HeadBundle, load_head, predict_proba
from .data import LabelMap
from .encoder import Encoder


class Stage1Predictor:
    def __init__(
        self,
        cfg: dict[str, Any],
        model_dir: Path | str,
        encoder: Encoder | None = None,
        bundle: HeadBundle | None = None,
    ):
        self.cfg = cfg
        model_dir = Path(model_dir)

        if bundle is None:
            classifier_path = model_dir / Path(cfg["paths"]["classifier"]).name
            bundle = load_head(classifier_path)
        self.bundle = bundle
        self.label_map: LabelMap = bundle.label_map

        if encoder is None:
            # SetFit models record a local fine-tuned encoder dir in meta; frozen
            # models record the HF repo id. Load whichever applies.
            enc_path = bundle.meta.get("encoder_path") or cfg["encoder"]["model_name"]
            if enc_path and Path(enc_path).exists():
                encoder = Encoder(cfg, model_name_or_path=enc_path)
            else:
                encoder = Encoder(cfg)
        self.encoder = encoder

    def predict_proba(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        """Return [N, C] calibrated probabilities aligned to label_map ids."""
        X = self.encoder.embed(list(texts), show_progress_bar=show_progress_bar)
        return predict_proba(self.bundle, X)

    @staticmethod
    def topk_indices(proba: np.ndarray, k: int) -> np.ndarray:
        """Return [N, k] intent ids per row, sorted by probability descending."""
        k = int(min(k, proba.shape[1]))
        # Partial top-k, then sort the k by prob desc.
        part = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(proba.shape[0])[:, None]
        order = np.argsort(-proba[rows, part], axis=1)
        return part[rows, order]
