"""Phase 0 smoke test: scaffold sanity only.

Confirms config loads, the ``src`` package and every stub sub-module import cleanly, and the config
loader resolves repo-relative paths. Per-module behavioural tests arrive with their phases
(normalization in Phase 1, cache lifecycle in Phase 3, etc.).
"""

from __future__ import annotations

import importlib

import pytest


# Every src module must import with only the core deps installed (stubs defer heavy/optional imports).
SRC_MODULES = [
    "src",
    "src.config",
    "src.preprocess.normalize",
    "src.embeddings.jina_client",
    "src.store.exact_store",
    "src.store.vector_store",
    "src.classifier.head",
    "src.classifier.combiner",
    "src.llm.fallback",
    "src.pipeline",
    "src.serve",
]


@pytest.mark.parametrize("module", SRC_MODULES)
def test_module_imports(module: str):
    importlib.import_module(module)


def test_config_loads_and_resolves_paths():
    from src.config import load_config, repo_path

    cfg = load_config()
    assert isinstance(cfg, dict) and cfg
    # repo_path resolves a config-relative path to a real location under the repo.
    assert repo_path(cfg["paths"]["intents"]).name == "intents.yaml"


@pytest.mark.parametrize(
    "module",
    [
        "scripts.seed_cache",
        "scripts.evaluate",
        "scripts.calibrate_thresholds",
        "scripts.simulate_growth",
        "scripts.audit_cache",
    ],
)
def test_scripts_import_and_expose_main(module: str):
    m = importlib.import_module(module)
    assert callable(getattr(m, "main"))
