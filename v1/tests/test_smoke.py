"""Phase 0 smoke test: scaffold sanity only.

Confirms config loads, required keys exist, the src package imports, and
the latency budget hasn't drifted from the design constant. Per-module
tests arrive with their respective phases.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_config_loads_and_has_required_keys():
    cfg = _load_config()
    # Top-level sections
    for key in (
        "encoder",
        "history",
        "approach",
        "setfit",
        "reranker",
        "data",
        "oos",
        "preprocess",
        "training",
        "latency",
        "paths",
        "runtime",
    ):
        assert key in cfg, f"missing top-level config key: {key}"

    # SetFit knobs live in config (no magic numbers in the CLI layer).
    for sk in ("num_epochs", "num_iterations", "batch_size"):
        assert sk in cfg["setfit"], f"missing setfit config key: {sk}"

    # Reranker (parallel second stage) knobs.
    for rk in ("enabled", "model_name", "top_k", "selective", "score_weight"):
        assert rk in cfg["reranker"], f"missing reranker config key: {rk}"
    assert cfg["reranker"]["enabled"] is False, "reranker should default to off"

    # Critical nested values
    assert cfg["encoder"]["model_name"] == "intfloat/multilingual-e5-small"
    assert cfg["encoder"]["max_seq_length"] in range(32, 129), "seq len out of expected range"
    assert cfg["history"]["num_turns"] in (1, 2), "history.num_turns must be 1 or 2 per CLAUDE.md §7"
    assert cfg["approach"] in ("frozen", "setfit")
    assert cfg["oos"]["method"] in ("threshold", "prototype", "energy")
    assert cfg["training"]["seed"] == 42
    assert cfg["data"]["columns"]["label"] == "expected_intent"


def test_latency_budget_sentinel():
    cfg = _load_config()
    # Hard constraint from CLAUDE.md §2 — if this changes, latency design must change with it.
    assert cfg["latency"]["budget_ms"] == 100


def test_src_package_imports():
    import src  # noqa: F401
