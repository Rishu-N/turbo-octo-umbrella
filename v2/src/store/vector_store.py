"""Vector store — semantic nearest-neighbor index over Jina embeddings.  [Phase 2; namespacing in P4]

The second cache layer (§1 step 3): on an exact-match miss, embed the normalized query (Jina) and find
the nearest labeled vector. Backend is hnswlib (config: vector_store.*) on cosine / inner-product over
L2-normalized vectors (so dot product == cosine; for both spaces hnswlib returns distance = 1 - sim, so
similarity = 1 - distance). A SQLite sidecar maps each integer id -> {intent, source, prev_intent, ts}.
Swappable for faiss behind this interface.

Namespacing (§3, Phase 4): every vector carries a ``prev_intent`` namespace ("" = the query-only path).
``query(..., prev_intent=X)`` only matches vectors stored under that same namespace, so a query-only
lookup can never match a context-conditioned ("yes do it" after card_lost) write-back, and vice versa.

Public surface:
    VectorStore(cfg)
      .add(vector, intent, source, prev_intent=None) -> int   # returns the assigned id
      .query(vector, k=1, prev_intent=None) -> list[Neighbor]  # nearest within the namespace
      .save() / .load()
      .size() -> int
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Literal

import hnswlib
import numpy as np

from src.config import repo_path

Source = Literal["seed", "writeback"]


@dataclass
class Neighbor:
    """A single nearest-neighbor result."""

    intent: str
    score: float          # cosine similarity in [-1, 1] (dot product on normalized vectors)
    source: Source
    id: int


class VectorStore:
    """hnswlib NN index + SQLite metadata sidecar, over L2-normalized embeddings."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        vs = cfg["vector_store"]
        self._dim = int(vs["dim"])
        self._space = vs.get("space", "cosine")
        self._max = int(vs["max_elements"])
        self._ef_construction = int(vs["ef_construction"])
        self._ef_search = int(vs["ef_search"])
        self._M = int(vs["M"])

        self._index_path = repo_path(vs["index_path"])
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path = repo_path(vs["meta_path"])
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._meta_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, intent TEXT NOT NULL, "
            "source TEXT NOT NULL, prev_intent TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)"
        )
        # Migration: add prev_intent to a pre-existing (Phase 2) meta table that lacks it.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        if "prev_intent" not in cols:
            self._conn.execute("ALTER TABLE meta ADD COLUMN prev_intent TEXT NOT NULL DEFAULT ''")
        self._conn.commit()

        self._index = hnswlib.Index(space=self._space, dim=self._dim)
        if self._index_path.exists():
            self._index.load_index(str(self._index_path), max_elements=self._max)
        else:
            self._index.init_index(
                max_elements=self._max, ef_construction=self._ef_construction, M=self._M
            )
        self._index.set_ef(self._ef_search)

        # In-memory id -> prev_intent namespace, for fast post-filtering of NN results.
        self._prev_by_id: dict[int, str] = {
            int(i): (p or "") for i, p in self._conn.execute("SELECT id, prev_intent FROM meta")
        }
        row = self._conn.execute("SELECT MAX(id) FROM meta").fetchone()
        self._next_id = (row[0] + 1) if row and row[0] is not None else 0

    def add(self, vector: np.ndarray, intent: str, source: Source, prev_intent: str | None = None) -> int:
        """Add a vector with its intent + source + namespace; return the assigned integer id."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self._dim:
            raise ValueError(f"vector dim {v.shape[0]} != index dim {self._dim}")
        if self._index.get_current_count() + 1 > self._max:
            self._max *= 2
            self._index.resize_index(self._max)
        new_id = self._next_id
        self._next_id += 1
        ns = prev_intent or ""
        self._index.add_items(v.reshape(1, -1), np.array([new_id]))
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(id, intent, source, prev_intent, created_at) VALUES (?, ?, ?, ?, ?)",
            (new_id, intent, source, ns, time.time()),
        )
        self._conn.commit()
        self._prev_by_id[new_id] = ns
        return new_id

    def query(self, vector: np.ndarray, k: int = 1, prev_intent: str | None = None) -> list[Neighbor]:
        """Return the ``k`` nearest neighbors *within the ``prev_intent`` namespace* (None -> '').

        Over-fetches then filters by namespace in Python — robust across hnswlib versions and fine for
        modest caches; swap to hnswlib's native ``filter=`` for very large indexes.
        """
        count = self._index.get_current_count()
        if count == 0:
            return []
        target = prev_intent or ""
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        fetch = min(count, max(k * 10, 50))
        labels, distances = self._index.knn_query(v, k=fetch)
        out: list[Neighbor] = []
        for label, dist in zip(labels[0], distances[0]):
            i = int(label)
            if self._prev_by_id.get(i, "") != target:
                continue
            row = self._conn.execute("SELECT intent, source FROM meta WHERE id = ?", (i,)).fetchone()
            if row is None:
                continue
            out.append(Neighbor(intent=row[0], score=1.0 - float(dist), source=row[1], id=i))
            if len(out) >= k:
                break
        return out

    def save(self) -> None:
        """Persist the index (vector_store.index_path); metadata is committed live."""
        self._index.save_index(str(self._index_path))

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if self._index_path.exists():
            self._index.load_index(str(self._index_path), max_elements=self._max)
            self._index.set_ef(self._ef_search)

    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.get_current_count()
