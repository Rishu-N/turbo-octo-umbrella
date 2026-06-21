"""Seed the cache: normalize + embed the seed data into the exact + vector stores.  [Phase 6]

Reads ``data/seed/seed.csv`` (config: paths.seed_csv), normalizes each query, embeds via Jina, and
writes entries with source="seed" (ground truth — never evicted), namespaced by the optional
``prev_intent`` column. Optionally fits + saves the learned head. Run once on the work laptop after
pointing config at your real labeled data (`export JINA_API_KEY=...`).

Usage:
    python scripts/seed_cache.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path  # noqa: E402
from src.preprocess.normalize import normalize  # noqa: E402


def load_seed_rows(path: str | Path) -> list[dict[str, str]]:
    """Load the seed CSV (columns: query, intent, lang, history, prev_intent)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_stores(cfg: dict[str, Any], embedder: Any, exact: Any, vector: Any, rows: list[dict]) -> dict:
    """Normalize + embed ``rows`` and write them as seed entries into both stores (namespaced)."""
    texts = [normalize(r["query"], cfg) for r in rows]
    vectors = embedder.embed(texts)
    written = 0
    for r, nq, vec in zip(rows, texts, vectors):
        prev = (r.get("prev_intent") or "").strip() or None
        result = exact.put(nq, r["intent"], "seed", prev_intent=prev)
        written += int(result.written)
        vector.add(vec, r["intent"], "seed", prev_intent=prev)
    vector.save()
    return {"rows": len(rows), "exact_written": written, "vectors": vector.size()}


def train_and_save_head(cfg: dict[str, Any], embedder: Any, rows: list[dict]) -> Any:
    """Fit the learned head on the seed embeddings and persist it to combiner.head_path."""
    from src.classifier.head import save_head, train_head

    texts = [normalize(r["query"], cfg) for r in rows]
    X = embedder.embed(texts)
    y = [r["intent"] for r in rows]
    bundle = train_head(X, y, cfg)
    save_head(bundle, str(repo_path(cfg["combiner"]["head_path"])))
    return bundle


def main() -> None:
    from src.embeddings.jina_client import JinaClient
    from src.store.exact_store import ExactStore
    from src.store.vector_store import VectorStore

    cfg = load_config()
    embedder = JinaClient(cfg)
    rows = load_seed_rows(repo_path(cfg["paths"]["seed_csv"]))
    print(seed_stores(cfg, embedder, ExactStore(cfg), VectorStore(cfg), rows))
    if cfg["combiner"]["mode"] != "nn_only":
        train_and_save_head(cfg, embedder, rows)
        print(f"trained + saved head -> {cfg['combiner']['head_path']}")


if __name__ == "__main__":
    main()
