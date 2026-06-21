"""Phase 2 tests: the SQLite exact-match store + write-back safety (src/store/exact_store.py)."""

from __future__ import annotations

import time

from src.store.exact_store import ExactStore


def _cfg(tmp_path, ttl_days=90, max_entries=50000, protect_seed=True, record_conflicts=True) -> dict:
    return {
        "exact_store": {"backend": "sqlite", "path": str(tmp_path / "exact.sqlite")},
        "eviction": {"ttl_days": ttl_days, "max_entries": max_entries, "protect_seed": protect_seed},
        "audit": {"record_conflicts": record_conflicts},
    }


def test_put_get_roundtrip(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    r = s.put("whats my balance", "balance_inquiry", "seed")
    assert r.written and not r.conflict
    e = s.get("whats my balance")
    assert e is not None and e.intent == "balance_inquiry" and e.source == "seed"


def test_miss_returns_none(tmp_path):
    assert ExactStore(_cfg(tmp_path)).get("nope") is None


def test_same_intent_is_idempotent(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    s.put("k", "i", "writeback", 0.9)
    r = s.put("k", "i", "writeback", 0.9)
    assert not r.written and not r.conflict and r.existing_intent == "i"


def test_seed_not_overwritten_by_writeback(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    s.put("k", "balance_inquiry", "seed")
    r = s.put("k", "card_lost_or_stolen", "writeback", 0.95)
    assert r.conflict and not r.written
    assert s.get("k").intent == "balance_inquiry"
    assert s.stats()["conflicts"] == 1


def test_writeback_conflict_not_flipflopped(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    s.put("k", "a", "writeback", 0.9)
    r = s.put("k", "b", "writeback", 0.95)
    assert r.conflict and not r.written
    assert s.get("k").intent == "a"


def test_seed_overrides_prior_writeback(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    s.put("k", "a", "writeback", 0.9)
    r = s.put("k", "b", "seed")
    assert r.written and r.conflict
    e = s.get("k")
    assert e.intent == "b" and e.source == "seed"


def test_prev_intent_namespacing(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    s.put("yes", "transfer_funds", "seed", prev_intent="transfer_funds")
    s.put("yes", "card_lost_or_stolen", "seed", prev_intent="card_lost_or_stolen")
    assert s.get("yes", prev_intent="transfer_funds").intent == "transfer_funds"
    assert s.get("yes", prev_intent="card_lost_or_stolen").intent == "card_lost_or_stolen"
    assert s.get("yes") is None  # the query-only ('' namespace) bucket is separate


def test_ttl_eviction_protects_seed(tmp_path):
    s = ExactStore(_cfg(tmp_path, ttl_days=1))
    s.put("old_wb", "i", "writeback", 0.9)
    s.put("old_seed", "j", "seed")
    s._conn.execute("UPDATE entries SET created_at = ?", (time.time() - 10 * 86400,))
    s._conn.commit()
    assert s.evict() == 1
    assert s.get("old_wb") is None
    assert s.get("old_seed") is not None  # seed protected from TTL


def test_size_cap_evicts_lru_writeback(tmp_path):
    s = ExactStore(_cfg(tmp_path, ttl_days=0, max_entries=2))
    for k in ("a", "b", "c"):
        s.put(k, "i", "writeback", 0.9)
        time.sleep(0.005)
    s.get("b")  # touch b and c so 'a' is least-recently-used
    s.get("c")
    assert s.evict() == 1
    assert s.get("a") is None
    assert s.get("b") is not None and s.get("c") is not None


def test_persistence_across_instances(tmp_path):
    cfg = _cfg(tmp_path)
    ExactStore(cfg).put("k", "i", "seed")
    assert ExactStore(cfg).get("k").intent == "i"


def test_conflict_recording_toggle(tmp_path):
    s = ExactStore(_cfg(tmp_path, record_conflicts=False))
    s.put("k", "a", "writeback", 0.9)
    s.put("k", "b", "writeback", 0.9)
    assert s.stats()["conflicts"] == 0


def test_stats_counts(tmp_path):
    s = ExactStore(_cfg(tmp_path))
    s.put("a", "balance_inquiry", "seed")
    s.put("b", "balance_inquiry", "writeback", 0.9)
    s.put("c", "card_lost_or_stolen", "seed")
    st = s.stats()
    assert st["total"] == 3
    assert st["by_source"] == {"seed": 2, "writeback": 1}
    assert st["per_intent"]["balance_inquiry"] == 2
