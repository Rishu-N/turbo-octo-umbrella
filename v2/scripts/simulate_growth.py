"""Self-improvement / error-amplification simulation (§12).  [Phase 6]

Replays a query stream through a deliberately NOISY LLM oracle (wrong on a fraction of queries, and less
confident when wrong) and measures, as the cache fills: cache hit-rate, end-to-end accuracy, and the
**cache false-hit rate** (wrong intent served from the cache — the error-amplification risk). Sweeping
``T_write`` shows the tradeoff: a higher write-back gate caches fewer answers (lower hit-rate) but admits
fewer errors (lower false-hit rate) — this is how T_write is chosen.

Offline (MockEmbedder) the self-improvement shows up through the exact-match cache on repeated queries;
real semantic generalization needs Jina embeddings.

Usage:
    python scripts/simulate_growth.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_config, repo_path  # noqa: E402
from src.llm.fallback import LLMClassifier, LLMResult  # noqa: E402

_CACHE_SOURCES = {"exact_cache", "semantic_cache", "head"}


class NoisyOracle(LLMClassifier):
    """A deliberately imperfect LLM: returns the gold intent w.p. (1 - noise), else a random wrong one.

    Mimics a real classifier's calibration loosely — high confidence when right, lower when wrong — so the
    ``T_write`` gate can filter some (but not all) errors.
    """

    def __init__(self, cfg: dict[str, Any], gold_by_query: dict[str, str], noise: float,
                 intents: list[str], seed: int = 42) -> None:
        self._gold = gold_by_query
        self._noise = float(noise)
        self._intents = intents
        self._fallback = cfg["oos"]["fallback_intent"]
        self._rng = np.random.default_rng(seed)

    def classify(self, query: str, history: list[str] | None = None) -> LLMResult:
        gold = self._gold.get(query, self._fallback)
        if self._rng.random() < self._noise:
            wrong = [i for i in self._intents if i != gold] or [gold]
            pred = wrong[int(self._rng.integers(len(wrong)))]
            return LLMResult(pred, float(self._rng.uniform(0.50, 0.90)))
        return LLMResult(gold, float(self._rng.uniform(0.85, 0.99)))


def make_stream(rows: list[dict], repeats: int = 5, seed: int = 42) -> list[dict]:
    """Build a replay stream by repeating the labeled rows and shuffling (deterministically)."""
    rng = np.random.default_rng(seed)
    stream = list(rows) * repeats
    rng.shuffle(stream)
    return stream


def simulate(cfg: dict[str, Any], embedder: Any, stream: list[dict], t_write: float,
             noise: float, seed: int = 42) -> dict:
    """Replay ``stream`` with a noisy oracle at the given ``t_write``; return growth/accuracy metrics."""
    import copy

    from src.pipeline import Pipeline
    from src.store.exact_store import ExactStore
    from src.store.vector_store import VectorStore

    c = copy.deepcopy(cfg)
    c["thresholds"]["t_write"] = float(t_write)
    d = Path(tempfile.mkdtemp())
    c["exact_store"]["path"] = str(d / "exact.sqlite")
    c["vector_store"]["index_path"] = str(d / "vs.bin")
    c["vector_store"]["meta_path"] = str(d / "meta.sqlite")
    c["audit"]["log_path"] = str(d / "audit.jsonl")

    intents = sorted({r["intent"] for r in stream})
    gold_by_query = {r["query"]: r["intent"] for r in stream}
    oracle = NoisyOracle(c, gold_by_query, noise, intents, seed)
    pipeline = Pipeline(c, jina=embedder, exact=ExactStore(c), vector=VectorStore(c), llm=oracle)

    n = correct = cache_hits = cache_false = 0
    hit_rate_series: list[float] = []
    for r in stream:
        prev = (r.get("prev_intent") or "").strip() or None
        res = pipeline.classify(r["query"], previous_intent=prev)
        n += 1
        correct += int(res.intent == r["intent"])
        if res.source in _CACHE_SOURCES:
            cache_hits += 1
            cache_false += int(res.intent != r["intent"])
        hit_rate_series.append(cache_hits / n)

    return {
        "t_write": float(t_write),
        "noise": float(noise),
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "hit_rate": cache_hits / n if n else 0.0,
        "cache_false_hit_rate": cache_false / cache_hits if cache_hits else 0.0,
        "hit_rate_series": hit_rate_series,
    }


def sweep_t_write(cfg: dict[str, Any], embedder: Any, stream: list[dict],
                  t_write_values: list[float], noise: float, seed: int = 42) -> list[dict]:
    """Run ``simulate`` across several T_write values (each with a fresh cache)."""
    return [simulate(cfg, embedder, stream, t, noise, seed) for t in t_write_values]


def main() -> None:
    import os

    from scripts.seed_cache import load_seed_rows

    cfg = load_config()
    rows = [r for r in load_seed_rows(repo_path(cfg["paths"]["seed_csv"])) if r["intent"] != cfg["oos"]["fallback_intent"]]
    stream = make_stream(rows, repeats=6)

    if os.environ.get("JINA_API_KEY"):
        from src.embeddings.jina_client import JinaClient

        embedder = JinaClient(cfg)
    else:
        from src.embeddings.mock import MockEmbedder

        embedder = MockEmbedder(cfg["vector_store"]["dim"])
        print("(no JINA_API_KEY -> MockEmbedder; self-improvement shown via the exact-match cache)\n")

    results = sweep_t_write(cfg, embedder, stream, [0.50, 0.70, 0.90, 0.95], noise=0.20)
    print(f"{'T_write':>8} {'hit_rate':>9} {'accuracy':>9} {'cache_false_hit':>16}")
    for r in results:
        print(f"{r['t_write']:>8.2f} {r['hit_rate']:>9.3f} {r['accuracy']:>9.3f} {r['cache_false_hit_rate']:>16.3f}")
    print("\nLower T_write -> more caching (higher hit-rate) but more amplified errors (higher false-hit).")


if __name__ == "__main__":
    main()
