"""Deterministic offline embedder for dev / scripts / tests — NOT for production.  [Phase 6 helper]

Produces stable pseudo-random unit vectors from a hash of the text: the same text always yields the
same vector, so the exact-match cache and repeat behavior are exercised offline. It has **no real
semantics**, so semantic-similarity numbers from it are meaningless — run with the real Jina client
(``JINA_API_KEY`` set) for meaningful calibration / evaluation. Exposes the same surface as
``JinaClient`` (``embed`` / ``embed_one``) so it is a drop-in for offline runs.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class MockEmbedder:
    """Hash-seeded deterministic unit-vector embedder (offline only)."""

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)
