"""Derive T_high / T_low from a sweep on held-out data — do NOT hardcode guesses.  [Phase 6]

Seeds a vector store with one split of the labeled data, then for a held-out split records the nearest-
neighbor similarity and whether the top match's intent is correct. Sweeping the acceptance threshold
yields hit-rate vs false-hit-rate curves; ``recommend`` picks the lowest threshold whose false-hit rate
is within a target (max coverage at acceptable risk). T_high uses query-only NN; T_low uses the
previous-intent-namespaced NN. (T_write is derived from the error-amplification tradeoff in
``simulate_growth.py``.)

Thresholds are Jina-v3-specific: any value from the literature is for a different model and MUST be
re-derived here on the work laptop with ``JINA_API_KEY`` set (the MockEmbedder gives degenerate curves).

Usage:
    python scripts/calibrate_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_config, repo_path  # noqa: E402
from src.preprocess.normalize import normalize  # noqa: E402

Score = tuple[float, str, str]  # (similarity, predicted_intent, gold_intent)


def sweep_thresholds(scores: list[Score], grid: list[float]) -> list[dict]:
    """For each threshold in ``grid``, compute hit-rate, false-hit-rate, and accuracy-on-hits."""
    n = len(scores)
    curve = []
    for t in grid:
        accepted = [(sim, pred, gold) for sim, pred, gold in scores if sim >= t]
        wrong = sum(1 for _, pred, gold in accepted if pred != gold)
        n_hit = len(accepted)
        curve.append(
            {
                "threshold": round(float(t), 4),
                "hit_rate": (n_hit / n) if n else 0.0,
                "false_hit_rate": (wrong / n_hit) if n_hit else 0.0,
                "accuracy_on_hits": ((n_hit - wrong) / n_hit) if n_hit else 0.0,
                "n_hits": n_hit,
            }
        )
    return curve


def recommend(curve: list[dict], max_false_hit: float) -> float:
    """Lowest threshold whose false-hit rate <= target (max coverage); else the most conservative."""
    ok = [row for row in curve if row["false_hit_rate"] <= max_false_hit and row["n_hits"] > 0]
    if ok:
        best = max(ok, key=lambda r: (r["hit_rate"], -r["threshold"]))
        return best["threshold"]
    return max(curve, key=lambda r: r["threshold"])["threshold"]


def collect_scores(
    cfg: dict[str, Any], embedder: Any, vector: Any, eval_rows: list[dict], namespaced: bool
) -> list[Score]:
    """Embed each eval query, take the top-1 neighbor (namespaced or not), and record (sim, pred, gold)."""
    scores: list[Score] = []
    for r in eval_rows:
        nq = normalize(r["query"], cfg)
        vec = embedder.embed_one(nq)
        prev = (r.get("prev_intent") or "").strip() or None if namespaced else None
        neighbors = vector.query(vec, k=1, prev_intent=prev)
        if neighbors:
            scores.append((float(neighbors[0].score), neighbors[0].intent, r["intent"]))
        else:
            scores.append((0.0, "", r["intent"]))
    return scores


def default_grid() -> list[float]:
    return [round(x, 2) for x in np.arange(0.50, 0.97, 0.02)]


def main() -> None:
    import json

    from scripts.seed_cache import load_seed_rows
    from src.embeddings.jina_client import JinaClient
    from src.store.vector_store import VectorStore

    cfg = load_config()
    rows = load_seed_rows(repo_path(cfg["paths"]["seed_csv"]))
    # Deterministic split: even-index rows seed, odd-index rows held out.
    seed_rows, eval_rows = rows[0::2], rows[1::2]
    embedder = JinaClient(cfg)

    # Seed a throwaway in-place store for calibration (uses configured paths; purge before a real run).
    vector = VectorStore(cfg)
    for r in seed_rows:
        prev = (r.get("prev_intent") or "").strip() or None
        vector.add(embedder.embed_one(normalize(r["query"], cfg)), r["intent"], "seed", prev_intent=prev)

    grid = default_grid()
    t_high_curve = sweep_thresholds(collect_scores(cfg, embedder, vector, eval_rows, namespaced=False), grid)
    t_low_curve = sweep_thresholds(collect_scores(cfg, embedder, vector, eval_rows, namespaced=True), grid)
    target = 0.05  # max acceptable false-hit rate
    print(json.dumps({"t_high_curve": t_high_curve, "t_low_curve": t_low_curve}, indent=2))
    print(f"\nRecommended (max_false_hit={target}):")
    print(f"  t_high = {recommend(t_high_curve, target)}")
    print(f"  t_low  = {recommend(t_low_curve, target)}")
    print("  t_write -> derive from scripts/simulate_growth.py (error-amplification tradeoff)")


if __name__ == "__main__":
    main()
