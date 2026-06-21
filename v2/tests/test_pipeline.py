"""Phase 3 tests: the query-only pipeline + LLM stub (src/pipeline.py, src/llm/fallback.py).

Jina is faked (deterministic vectors, or forced failure); stores are real but write to tmp_path.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from src.config import load_config
from src.embeddings.jina_client import JinaAPIError
from src.llm.fallback import StubLLMClassifier
from src.pipeline import Pipeline
from src.preprocess.normalize import normalize


def _cfg(tmp_path) -> dict:
    cfg = load_config()
    cfg["exact_store"]["path"] = str(tmp_path / "exact.sqlite")
    cfg["vector_store"]["index_path"] = str(tmp_path / "vs.bin")
    cfg["vector_store"]["meta_path"] = str(tmp_path / "meta.sqlite")
    cfg["audit"]["log_path"] = str(tmp_path / "audit.jsonl")
    return cfg


class FakeJina:
    """Returns injected vectors for known texts; deterministic random unit vectors otherwise."""

    def __init__(self, dim: int, vectors: dict[str, np.ndarray] | None = None, fail: bool = False):
        self.dim = dim
        self.vectors = vectors or {}
        self.fail = fail
        self.calls = 0

    def embed_one(self, text: str) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise JinaAPIError("down")
        if text in self.vectors:
            return self.vectors[text]
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        v = np.random.default_rng(seed).standard_normal(self.dim).astype("float32")
        return v / np.linalg.norm(v)


def _pipeline(cfg, jina=None, mapping=None):
    dim = cfg["vector_store"]["dim"]
    return Pipeline(
        cfg,
        jina=jina or FakeJina(dim),
        llm=StubLLMClassifier(cfg, mapping=mapping),
    )


# --- the three cache outcomes ------------------------------------------------------------------
def test_exact_cache_hit(tmp_path):
    cfg = _cfg(tmp_path)
    p = _pipeline(cfg)
    p._exact.put(normalize("what is my balance", cfg), "balance_inquiry", "seed")
    r = p.classify("What is my balance?")
    assert r.source == "exact_cache" and r.intent == "balance_inquiry" and not r.is_oos


def test_semantic_cache_hit(tmp_path):
    cfg = _cfg(tmp_path)
    dim = cfg["vector_store"]["dim"]
    v = np.zeros(dim, dtype="float32")
    v[0] = 1.0
    jina = FakeJina(dim, vectors={normalize("send money to john", cfg): v})
    p = _pipeline(cfg, jina=jina)
    p._vector.add(v, "transfer_funds", "seed")
    r = p.classify("send money to john")
    assert r.source == "semantic_cache" and r.intent == "transfer_funds"
    assert r.confidence >= cfg["thresholds"]["t_high"]


# --- the miss -> write-back -> hit lifecycle ----------------------------------------------------
def test_miss_then_writeback_then_exact_hit(tmp_path):
    cfg = _cfg(tmp_path)
    p = _pipeline(cfg, mapping={"track my loan application": ("loan_inquiry", 0.97)})
    r1 = p.classify("track my loan application")
    assert r1.source == "llm_fallback" and r1.intent == "loan_inquiry"
    assert p._vector.size() == 1  # high-confidence -> written back to the vector store too
    r2 = p.classify("track my loan application")
    assert r2.source == "exact_cache" and r2.intent == "loan_inquiry"


# --- write-back safety gate ---------------------------------------------------------------------
def test_low_confidence_is_served_but_not_cached(tmp_path):
    cfg = _cfg(tmp_path)
    p = _pipeline(cfg, mapping={"some borderline ask": ("balance_inquiry", 0.70)})  # >= oos_floor, < T_write
    r1 = p.classify("some borderline ask")
    assert r1.source == "llm_fallback" and r1.intent == "balance_inquiry" and not r1.is_oos
    r2 = p.classify("some borderline ask")
    assert r2.source == "llm_fallback"  # still a miss -> the gate refused to cache it
    assert p._vector.size() == 0
    assert p._exact.get(normalize("some borderline ask", cfg)) is None


def test_out_of_scope_returns_fallback(tmp_path):
    cfg = _cfg(tmp_path)
    p = _pipeline(cfg, mapping={"qwerty asdf zxcv": ("balance_inquiry", 0.20)})  # < oos_floor
    r = p.classify("qwerty asdf zxcv")
    assert r.is_oos and r.intent == cfg["oos"]["fallback_intent"] and r.source == "llm_fallback"
    assert p._vector.size() == 0  # OOS is never written back


def test_api_down_skips_semantic_and_writeback(tmp_path):
    cfg = _cfg(tmp_path)
    dim = cfg["vector_store"]["dim"]
    p = _pipeline(cfg, jina=FakeJina(dim, fail=True),
                  mapping={"my card was declined": ("card_payment_declined", 0.99)})
    r = p.classify("my card was declined")
    assert r.source == "llm_fallback" and r.intent == "card_payment_declined"
    # API down -> no embedding -> no write-back even though confidence >= T_write
    assert p._vector.size() == 0
    assert p._exact.get(normalize("my card was declined", cfg)) is None


# --- audit + stub ------------------------------------------------------------------------------
def test_writeback_is_audited(tmp_path):
    cfg = _cfg(tmp_path)
    p = _pipeline(cfg, mapping={"open a fixed deposit": ("loan_inquiry", 0.95)})
    p.classify("open a fixed deposit")
    lines = pathlib.Path(cfg["audit"]["log_path"]).read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["intent"] == "loan_inquiry" and rec["written"] is True and rec["source"] == "writeback"


def test_stub_keyword_rules_and_oos(tmp_path):
    cfg = _cfg(tmp_path)
    stub = StubLLMClassifier(cfg)
    assert stub.classify("what is my balance").intent == "balance_inquiry"
    assert stub.classify("i lost my card").intent == "card_lost_or_stolen"
    miss = stub.classify("zzz total nonsense zzz")
    assert miss.intent == cfg["oos"]["fallback_intent"]
    assert miss.confidence < cfg["llm_fallback"]["oos_confidence_floor"]


# --- Phase 5: the learned-head stage in the pipeline -------------------------------------------
def _onehot(dim, i):
    v = np.zeros(dim, dtype="float32")
    v[i] = 1.0
    return v


def _trained_head(cfg):
    from src.classifier.head import train_head

    dim = cfg["vector_store"]["dim"]
    e0, e1 = _onehot(dim, 0), _onehot(dim, 1)
    X = np.vstack([e0] * 5 + [e1] * 5)
    y = ["balance_inquiry"] * 5 + ["transfer_funds"] * 5
    return train_head(X, y, cfg), e0


def test_head_stage_resolves_a_cache_miss(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["combiner"]["mode"] = "head_only"
    cfg["combiner"]["calibration"] = "none"
    cfg["combiner"]["head_threshold"] = 0.5
    bundle, e0 = _trained_head(cfg)
    dim = cfg["vector_store"]["dim"]
    jina = FakeJina(dim, {normalize("novel balance phrasing here", cfg): e0})
    p = Pipeline(cfg, jina=jina, llm=StubLLMClassifier(cfg), head=bundle)
    r = p.classify("novel balance phrasing here")
    assert r.source == "head" and r.intent == "balance_inquiry"


def test_nn_only_mode_skips_the_head(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["combiner"]["mode"] = "nn_only"
    cfg["combiner"]["calibration"] = "none"
    bundle, e0 = _trained_head(cfg)
    dim = cfg["vector_store"]["dim"]
    jina = FakeJina(dim, {normalize("novel balance phrasing here", cfg): e0})
    p = Pipeline(cfg, jina=jina, llm=StubLLMClassifier(cfg), head=bundle)
    r = p.classify("novel balance phrasing here")
    assert r.source == "llm_fallback"  # head disabled in nn_only -> LLM handles it
