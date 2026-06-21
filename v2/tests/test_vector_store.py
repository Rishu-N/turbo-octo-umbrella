"""Phase 2 tests: the hnswlib vector store + SQLite metadata (src/store/vector_store.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.store.vector_store import VectorStore


def _cfg(tmp_path, dim=8, max_elements=10) -> dict:
    return {
        "vector_store": {
            "backend": "hnswlib",
            "space": "cosine",
            "dim": dim,
            "max_elements": max_elements,
            "ef_construction": 100,
            "ef_search": 10,
            "M": 16,
            "index_path": str(tmp_path / "vs.bin"),
            "meta_path": str(tmp_path / "vs_meta.sqlite"),
        }
    }


def _unit(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _onehot(i, dim=8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_add_and_query_nearest(tmp_path):
    s = VectorStore(_cfg(tmp_path))
    s.add(_onehot(0), "balance_inquiry", "seed")
    s.add(_onehot(1), "card_lost_or_stolen", "seed")
    res = s.query(_onehot(0), k=1)
    assert len(res) == 1
    assert res[0].intent == "balance_inquiry"
    assert res[0].score > 0.99  # cosine similarity of identical unit vectors ~= 1


def test_query_empty_index(tmp_path):
    assert VectorStore(_cfg(tmp_path)).query(_onehot(0)) == []


def test_size(tmp_path):
    s = VectorStore(_cfg(tmp_path))
    assert s.size() == 0
    s.add(_onehot(0), "i", "seed")
    assert s.size() == 1


def test_k_is_clamped_to_count(tmp_path):
    s = VectorStore(_cfg(tmp_path))
    s.add(_onehot(0), "i", "seed")
    assert len(s.query(_onehot(0), k=5)) == 1


def test_dim_mismatch_raises(tmp_path):
    s = VectorStore(_cfg(tmp_path, dim=8))
    with pytest.raises(ValueError):
        s.add(np.ones(4, dtype=np.float32), "i", "seed")


def test_neighbor_carries_metadata(tmp_path):
    s = VectorStore(_cfg(tmp_path))
    s.add(_onehot(2), "refund_status", "writeback")
    n = s.query(_onehot(2), k=1)[0]
    assert n.intent == "refund_status" and n.source == "writeback" and isinstance(n.id, int)


def test_save_load_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    s = VectorStore(cfg)
    s.add(_onehot(0), "balance_inquiry", "seed")
    s.save()
    reopened = VectorStore(cfg)  # __init__ loads the persisted index
    res = reopened.query(_onehot(0), k=1)
    assert reopened.size() == 1
    assert res and res[0].intent == "balance_inquiry"


def test_resize_beyond_initial_max(tmp_path):
    s = VectorStore(_cfg(tmp_path, max_elements=2))
    for i in range(5):
        s.add(_unit(_onehot(i % 8) + 0.01 * _onehot((i + 1) % 8)), f"intent{i}", "writeback")
    assert s.size() == 5  # grew past the initial cap without error
