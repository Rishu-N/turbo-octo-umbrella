"""Phase 4 tests: previous-intent escalation, namespacing, adaptive T_low, window fusion."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.config import load_config
from src.embeddings.jina_client import JinaAPIError
from src.llm.fallback import StubLLMClassifier
from src.pipeline import Pipeline
from src.preprocess.normalize import normalize


def _cfg(tmp_path, sub="a") -> dict:
    cfg = load_config()
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    cfg["exact_store"]["path"] = str(d / "exact.sqlite")
    cfg["vector_store"]["index_path"] = str(d / "vs.bin")
    cfg["vector_store"]["meta_path"] = str(d / "meta.sqlite")
    cfg["audit"]["log_path"] = str(d / "audit.jsonl")
    return cfg


def _vec(dim, comps) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    for i, x in comps.items():
        v[i] = float(x)
    return (v / np.linalg.norm(v)).astype("float32")


class FakeJina:
    def __init__(self, dim, vectors=None, fail=False):
        self.dim = dim
        self.vectors = vectors or {}
        self.fail = fail

    def embed_one(self, text):
        if self.fail:
            raise JinaAPIError("down")
        if text in self.vectors:
            return self.vectors[text]
        s = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        v = np.random.default_rng(s).standard_normal(self.dim).astype("float32")
        return v / np.linalg.norm(v)


def _p(cfg, jina=None, mapping=None):
    dim = cfg["vector_store"]["dim"]
    return Pipeline(cfg, jina=jina or FakeJina(dim), llm=StubLLMClassifier(cfg, mapping=mapping))


# --- the headline example: same surface query, different context --------------------------------
def test_yes_do_it_resolved_by_previous_intent(tmp_path):
    cfg = _cfg(tmp_path)
    p = _p(cfg)
    nq = normalize("yes do it", cfg)
    p._exact.put(nq, "card_lost_or_stolen", "seed", prev_intent="card_lost_or_stolen")
    p._exact.put(nq, "transfer_funds", "seed", prev_intent="transfer_funds")

    assert p.classify("yes do it", previous_intent="card_lost_or_stolen").intent == "card_lost_or_stolen"
    assert p.classify("yes do it", previous_intent="transfer_funds").intent == "transfer_funds"
    # Without context the bare query is unresolved (query-only namespace is empty) -> LLM -> OOS.
    r = p.classify("yes do it")
    assert r.source == "llm_fallback" and r.is_oos


def test_query_only_does_not_match_namespaced_vector(tmp_path):
    cfg = _cfg(tmp_path)
    dim = cfg["vector_store"]["dim"]
    e0 = _vec(dim, {0: 1.0})
    p = _p(cfg, jina=FakeJina(dim, {normalize("the second one", cfg): e0}))
    p._vector.add(e0, "transfer_funds", "seed", prev_intent="transfer_funds")

    # Query-only must NOT see a context-namespaced vector.
    assert p.classify("the second one").source == "llm_fallback"
    # With the matching previous intent it does.
    r = p.classify("the second one", previous_intent="transfer_funds")
    assert r.source == "semantic_cache" and r.intent == "transfer_funds"


# --- adaptive T_low ----------------------------------------------------------------------------
def test_effective_t_low_is_adaptive(tmp_path):
    cfg = _cfg(tmp_path)
    p = _p(cfg)
    assert p._effective_t_low(0.0) == pytest.approx(cfg["thresholds"]["t_low"])
    assert p._effective_t_low(1.0) == pytest.approx(cfg["thresholds"]["t_low_floor"])
    assert cfg["thresholds"]["t_low_floor"] < p._effective_t_low(0.5) < cfg["thresholds"]["t_low"]


def test_escalation_accepts_at_or_above_t_low(tmp_path):
    cfg = _cfg(tmp_path)  # t_low ~0.78
    dim = cfg["vector_store"]["dim"]
    q = _vec(dim, {0: 1.0})
    stored = _vec(dim, {0: 0.80, 1: 0.60})  # cosine 0.80 with q
    p = _p(cfg, jina=FakeJina(dim, {normalize("what about for savings", cfg): q}))
    p._vector.add(stored, "balance_inquiry", "seed", prev_intent="balance_inquiry")
    r = p.classify("what about for savings", previous_intent="balance_inquiry")
    assert r.source == "semantic_cache" and r.intent == "balance_inquiry"
    assert r.confidence == pytest.approx(0.80, abs=0.02)


def test_escalation_rejects_below_t_low(tmp_path):
    cfg = _cfg(tmp_path)
    dim = cfg["vector_store"]["dim"]
    q = _vec(dim, {0: 1.0})
    stored = _vec(dim, {0: 0.70, 1: 0.7141})  # cosine 0.70 < t_low 0.78
    p = _p(cfg, jina=FakeJina(dim, {normalize("the second one", cfg): q}))
    p._vector.add(stored, "transfer_funds", "seed", prev_intent="transfer_funds")
    r = p.classify("the second one", previous_intent="transfer_funds")
    assert r.source == "llm_fallback"  # 0.70 fails the prev-intent-only T_low


def test_window_fusion_lowers_threshold_to_accept(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["history"]["mode"] = "prev_intent_window"
    cfg["history"]["window_turns"] = 2
    dim = cfg["vector_store"]["dim"]
    q = _vec(dim, {0: 1.0})
    stored = _vec(dim, {0: 0.70, 1: 0.7141})  # 0.70: below T_low, above the floor (0.65)
    nq = normalize("the second one", cfg)
    window = normalize("show my payees", cfg) + " [SEP] " + normalize("you have two payees", cfg)
    # The window embeds to q too, so fusion is identity and only the threshold-lowering matters.
    p = _p(cfg, jina=FakeJina(dim, {nq: q, window: q}))
    p._vector.add(stored, "transfer_funds", "seed", prev_intent="transfer_funds")
    r = p.classify(
        "the second one",
        history=["show my payees", "you have two payees"],
        previous_intent="transfer_funds",
    )
    assert r.source == "semantic_cache" and r.intent == "transfer_funds"


# --- gating & namespaced write-back -------------------------------------------------------------
def test_history_off_disables_escalation(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["history"]["mode"] = "off"
    p = _p(cfg)
    p._exact.put(normalize("yes do it", cfg), "card_lost_or_stolen", "seed", prev_intent="card_lost_or_stolen")
    assert p.classify("yes do it", previous_intent="card_lost_or_stolen").source == "llm_fallback"


def test_referential_writeback_is_namespaced(tmp_path):
    cfg = _cfg(tmp_path)
    p = _p(cfg, mapping={"yes do it": ("card_lost_or_stolen", 0.95)})
    r = p.classify("yes do it", previous_intent="card_lost_or_stolen")
    assert r.source == "llm_fallback" and r.intent == "card_lost_or_stolen"
    nq = normalize("yes do it", cfg)
    assert p._exact.get(nq, prev_intent="card_lost_or_stolen") is not None  # stored namespaced
    assert p._exact.get(nq) is None  # NOT in the query-only namespace
    assert p.classify("yes do it", previous_intent="card_lost_or_stolen").source == "exact_cache"


def test_self_contained_writeback_is_query_only(tmp_path):
    cfg = _cfg(tmp_path)
    long_q = "please help me apply for a personal home loan today"  # > referential_max_tokens
    p = _p(cfg, mapping={long_q: ("loan_inquiry", 0.95)})
    p.classify(long_q, previous_intent="balance_inquiry")
    nq = normalize(long_q, cfg)
    assert p._exact.get(nq) is not None  # query-only namespace (self-contained)
    assert p._exact.get(nq, prev_intent="balance_inquiry") is None
