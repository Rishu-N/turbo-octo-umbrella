"""Jina v3 embeddings API client — the ONLY network dependency on the hot path.  [Phase 2]

CRITICAL: jina-embeddings-v3 is API-only here (no local weights). Every embed() is a remote call.
This module isolates that call so it can be swapped, mocked in tests, and hardened with timeouts /
retries / graceful failure (if the API is down, ``embed`` raises ``JinaAPIError`` and the caller falls
straight through to the LLM fallback and skips write-back, §2).

Design (config: jina.*):
    - task adapter:  "classification" (intent-tuned LoRA)   -> request field ``task``
    - Matryoshka:    dimensions=256 (configurable)          -> request field ``dimensions``
    - normalize:     L2-normalize so cosine == dot product
    - embed cache:   persistent (SQLite), keyed by normalized text + model/task/dim/normalize, so we
                     never pay the API twice for the same string.
    - batching:      uncached inputs are sent in batches of ``batch_size`` to amortize round-trips.

Request shape (POST {jina.api_url}, header ``Authorization: Bearer $JINA_API_KEY``):
    {"model": "jina-embeddings-v3", "task": "classification", "dimensions": 256, "input": ["...", ...]}
Response shape:
    {"data": [{"embedding": [...], "index": 0}, ...], "usage": {...}}

Retries: a manual exponential-backoff loop (chosen over a urllib3 Retry adapter so transient vs fatal is
explicit and "skip write-back on failure" is clean to test). Retries on connection errors and HTTP
429/5xx; other 4xx are fatal.

HONESTY: only the local exact-match layer is sub-millisecond. A semantic-layer hit costs one Jina
round-trip (tens of ms+). Do NOT claim sub-10ms for this path.

Public surface:
    JinaClient(cfg).embed(texts) -> np.ndarray    # [N, dim], L2-normalized, cache-backed
    JinaClient(cfg).embed_one(text) -> np.ndarray # [dim]
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Sequence

import numpy as np
import requests

from src.config import repo_path


class JinaAPIError(RuntimeError):
    """Raised when the Jina API is unreachable or returns a fatal error after retries.

    The pipeline catches this, serves the LLM fallback, and skips write-back (§2).
    """


class _Transient(Exception):
    """Internal: a retryable failure (connection error or HTTP 429/5xx)."""


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; zero rows are left as zeros."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class JinaClient:
    """Remote Jina v3 embedder with a persistent, normalized-text-keyed embed cache."""

    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, cfg: dict[str, Any], session: requests.Session | None = None) -> None:
        self._cfg = cfg
        jc = cfg["jina"]
        self._url: str = jc["api_url"]
        self._model: str = jc["model"]
        self._task: str = jc["task"]
        self._dim: int = int(jc["dimensions"])
        self._normalize: bool = bool(jc.get("normalize", True))
        self._timeout: float = float(jc.get("timeout_seconds", 5.0))
        self._max_retries: int = int(jc.get("max_retries", 3))
        self._backoff: float = float(jc.get("backoff_base_seconds", 0.5))
        self._batch_size: int = int(jc.get("batch_size", 32))
        # Key from env only — never from config (§3 of the hard constraints).
        self._api_key: str | None = os.environ.get("JINA_API_KEY")
        self._session = session or requests.Session()

        cache_path = repo_path(jc["embed_cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(str(cache_path))
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS embed_cache "
            "(key TEXT PRIMARY KEY, vector BLOB NOT NULL, created_at REAL NOT NULL)"
        )
        self._cache.commit()

    # --- public API ------------------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an [N, dim] array of (optionally L2-normalized) embeddings, cache-backed + batched."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, self._dim), dtype=np.float32)

        out: list[np.ndarray | None] = [None] * n
        miss_order: list[str] = []
        miss_positions: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                out[i] = cached
            else:
                if t not in miss_positions:
                    miss_positions[t] = []
                    miss_order.append(t)
                miss_positions[t].append(i)

        for start in range(0, len(miss_order), self._batch_size):
            batch = miss_order[start : start + self._batch_size]
            vecs = np.asarray(self._request(batch), dtype=np.float32)
            if self._normalize:
                vecs = _l2_normalize(vecs)
            for t, v in zip(batch, vecs):
                self._cache_put(t, v)
                for i in miss_positions[t]:
                    out[i] = v

        return np.vstack(out).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Return a single [dim] embedding (cache-backed)."""
        return self.embed([text])[0]

    # --- network ---------------------------------------------------------------------------------
    def _request(self, batch: list[str]) -> list[list[float]]:
        """POST one batch to the Jina API with retry/backoff; raise JinaAPIError on fatal failure."""
        if not self._api_key:
            raise JinaAPIError("JINA_API_KEY is not set in the environment")
        payload = {
            "model": self._model,
            "task": self._task,
            "dimensions": self._dim,
            "input": list(batch),
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._session.post(
                    self._url, json=payload, headers=headers, timeout=self._timeout
                )
            except requests.exceptions.RequestException as e:  # connection error / timeout -> retryable
                last_exc = e
                if attempt < self._max_retries:
                    time.sleep(self._backoff * (2**attempt))
                    continue
                raise JinaAPIError(
                    f"Jina API request failed after {self._max_retries + 1} attempts: {e}"
                ) from e

            status = resp.status_code
            if status in self._RETRYABLE_STATUS:  # 429 / 5xx -> retryable
                last_exc = _Transient(f"HTTP {status}")
                if attempt < self._max_retries:
                    time.sleep(self._backoff * (2**attempt))
                    continue
                raise JinaAPIError(
                    f"Jina API returned HTTP {status} after {self._max_retries + 1} attempts"
                )
            if status >= 400:  # other 4xx -> fatal, do NOT retry
                raise JinaAPIError(f"Jina API returned fatal HTTP {status}")

            try:
                data = resp.json()["data"]
            except (KeyError, ValueError, TypeError) as e:  # malformed response body
                raise JinaAPIError(f"Unexpected Jina API response: {e}") from e
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in ordered]
        raise JinaAPIError(f"Jina API failed: {last_exc}")  # pragma: no cover (loop returns/raises)

    # --- cache -----------------------------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        return f"{self._model}|{self._task}|{self._dim}|{int(self._normalize)}|{text}"

    def _cache_get(self, text: str) -> np.ndarray | None:
        row = self._cache.execute(
            "SELECT vector FROM embed_cache WHERE key = ?", (self._cache_key(text),)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def _cache_put(self, text: str, vector: np.ndarray) -> None:
        self._cache.execute(
            "INSERT OR REPLACE INTO embed_cache(key, vector, created_at) VALUES (?, ?, ?)",
            (self._cache_key(text), vector.astype(np.float32).tobytes(), time.time()),
        )
        self._cache.commit()
