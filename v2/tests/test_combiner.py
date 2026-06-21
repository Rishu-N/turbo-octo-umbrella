"""Phase 5 tests: the NN/head combiner (src/classifier/combiner.py)."""

from __future__ import annotations

import pytest

from src.classifier.combiner import Combiner
from src.config import load_config
from src.store.vector_store import Neighbor


def _cfg(mode, threshold=0.5):
    cfg = load_config()
    cfg["combiner"]["mode"] = mode
    cfg["combiner"]["head_threshold"] = threshold
    return cfg


def _nbr(intent, score):
    return Neighbor(intent=intent, score=score, source="seed", id=1)


def test_nn_only_defers_to_llm():
    assert Combiner(_cfg("nn_only")).decide([], {"balance_inquiry": 0.9}) is None


def test_no_head_proba_defers():
    assert Combiner(_cfg("head_only")).decide([], None) is None


def test_head_only_accepts_above_threshold():
    d = Combiner(_cfg("head_only", 0.5)).decide([], {"balance_inquiry": 0.8, "transfer_funds": 0.2})
    assert d is not None and d.intent == "balance_inquiry" and d.source == "head"
    assert d.confidence == pytest.approx(0.8) and not d.is_oos


def test_head_only_rejects_below_threshold():
    assert Combiner(_cfg("head_only", 0.6)).decide([], {"a": 0.4, "b": 0.35, "c": 0.25}) is None


def test_cascade_nn_agreement_reinforces_confidence():
    d = Combiner(_cfg("nn_then_head", 0.5)).decide(
        [_nbr("balance_inquiry", 0.84)], {"balance_inquiry": 0.55, "x": 0.45}
    )
    assert d.intent == "balance_inquiry" and d.confidence == pytest.approx(0.84)  # max(0.55, 0.84)


def test_cascade_disagreeing_nn_does_not_boost():
    d = Combiner(_cfg("nn_then_head", 0.5)).decide(
        [_nbr("transfer_funds", 0.99)], {"balance_inquiry": 0.6, "transfer_funds": 0.4}
    )
    assert d.intent == "balance_inquiry" and d.confidence == pytest.approx(0.6)


def test_cascade_below_threshold_defers():
    d = Combiner(_cfg("nn_then_head", 0.5)).decide(
        [_nbr("balance_inquiry", 0.45)], {"balance_inquiry": 0.45, "x": 0.30, "y": 0.25}
    )
    assert d is None
