"""Config sentinel tests: cross-cutting invariants that, if broken, break the design.

These guard the hard constraints from the brief so a careless config edit fails fast in CI rather
than silently degrading behaviour at runtime.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _cfg() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_top_level_sections_present():
    cfg = _cfg()
    for key in (
        "jina",
        "thresholds",
        "normalization",
        "history",
        "combiner",
        "vector_store",
        "exact_store",
        "eviction",
        "audit",
        "llm_fallback",
        "oos",
        "paths",
        "seed",
    ):
        assert key in cfg, f"missing top-level config key: {key}"


def test_jina_classification_adapter():
    # §2: use the classification task adapter; L2-normalize so cosine == dot product.
    jina = _cfg()["jina"]
    assert jina["task"] == "classification"
    assert jina["model"] == "jina-embeddings-v3"
    assert jina["normalize"] is True


def test_embedding_dim_matches_vector_store():
    # Hard invariant: the index dim MUST equal the requested Matryoshka dim, or NN search breaks.
    cfg = _cfg()
    assert cfg["jina"]["dimensions"] == cfg["vector_store"]["dim"]


def test_thresholds_are_valid_placeholders():
    # §12: thresholds are PLACEHOLDERS to be derived by calibrate_thresholds.py (Phase 6),
    # but must at least be valid probabilities/similarities with the floor below the adaptive T_low.
    t = _cfg()["thresholds"]
    for k in ("t_high", "t_low", "t_low_floor", "t_write"):
        assert 0.0 < t[k] < 1.0, f"{k} must be in (0, 1)"
    assert t["t_low_floor"] <= t["t_low"], "t_low_floor must not exceed t_low"


def test_history_mode_valid():
    assert _cfg()["history"]["mode"] in ("off", "prev_intent", "prev_intent_window")


def test_combiner_mode_valid():
    c = _cfg()["combiner"]
    assert c["mode"] in ("nn_only", "head_only", "nn_then_head")
    assert c["head_type"] in ("logreg", "nearest_centroid")


def test_seed_is_42():
    assert _cfg()["seed"] == 42


def test_fallback_intent_declared_in_taxonomy():
    # §8: the OOS/fallback intent must actually exist in the taxonomy.
    cfg = _cfg()
    fb = cfg["oos"]["fallback_intent"]
    with (REPO_ROOT / cfg["paths"]["intents"]).open() as f:
        intents = yaml.safe_load(f)
    assert fb in intents["intents"], "fallback intent must be declared in data/intents.yaml"


def test_no_api_key_in_config():
    # §2 / §13: the Jina key must come from the JINA_API_KEY env var, never the config file.
    assert "api_key" not in _cfg()["jina"], "API key must come from JINA_API_KEY env, not config"
