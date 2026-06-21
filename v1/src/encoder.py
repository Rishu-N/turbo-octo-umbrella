"""Sentence encoder wrapper.

Approach A treats this as frozen. Approach B (SetFit) fine-tunes the underlying
SentenceTransformer; after training, you can load the fine-tuned checkpoint
through the same `Encoder` to keep the inference path identical.

Backends:
    - pytorch     (default; what we use for training and current inference)
    - onnx_fp32   (Phase 3 — placeholder)
    - onnx_int8   (Phase 3 — placeholder)

The public surface is just `embed(texts) -> np.ndarray`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(
        self,
        cfg: dict[str, Any],
        model_name_or_path: str | Path | None = None,
        device: str | None = None,
    ):
        backend = cfg["encoder"].get("backend", "pytorch")
        if backend != "pytorch":
            # Phase 3 wires up ONNX backends. Until then we error loudly rather
            # than silently fall back, so config typos don't go unnoticed.
            raise NotImplementedError(
                f"encoder.backend={backend!r} is not implemented yet (Phase 3). "
                "Set encoder.backend: pytorch for now."
            )

        self.cfg = cfg
        self.model_name = str(model_name_or_path or cfg["encoder"]["model_name"])
        self.max_seq_length = int(cfg["encoder"]["max_seq_length"])
        self.model = SentenceTransformer(self.model_name, device=device)
        # Cap the tokenizer max_seq_length so the encoder truncates inputs.
        self.model.max_seq_length = self.max_seq_length

    def embed(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a list of strings into a float32 [N, dim] matrix."""
        embs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embs.astype(np.float32, copy=False)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
