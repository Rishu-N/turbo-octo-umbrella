"""Evaluation metrics (§12).  [Phase 6]

Replays a labeled set through the pipeline and reports: cache metrics (hit-rate, exact/semantic/head
split, false-hit rate = wrong intent served on a hit, latency per path), intent metrics (accuracy,
macro-F1), per-language breakdown (en/hi/hinglish), and OOS/fallback precision & recall.

Numbers are meaningful only with real Jina embeddings (JINA_API_KEY); offline (MockEmbedder) the
semantic layer is degenerate. Run on the work laptop against a held-out split.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import load_config, repo_path  # noqa: E402

_CACHE_SOURCES = {"exact_cache", "semantic_cache", "head"}


def evaluate_pipeline(cfg: dict[str, Any], pipeline: Any, rows: list[dict]) -> dict:
    """Replay ``rows`` (query, intent, lang, prev_intent) through ``pipeline`` and compute metrics."""
    fallback = cfg["oos"]["fallback_intent"]
    total = len(rows)
    correct = hits = false_hits = 0
    by_source: dict[str, int] = defaultdict(int)
    latency: dict[str, list[float]] = defaultdict(list)
    lang_tot: dict[str, int] = defaultdict(int)
    lang_correct: dict[str, int] = defaultdict(int)
    oos_tp = oos_fp = oos_fn = 0
    y_true: list[str] = []
    y_pred: list[str] = []

    for r in rows:
        prev = (r.get("prev_intent") or "").strip() or None
        t0 = time.perf_counter()
        res = pipeline.classify(r["query"], previous_intent=prev)
        latency[res.source].append((time.perf_counter() - t0) * 1000.0)

        gold = r["intent"]
        y_true.append(gold)
        y_pred.append(res.intent)
        ok = res.intent == gold
        correct += int(ok)
        by_source[res.source] += 1
        lang = r.get("lang", "?")
        lang_tot[lang] += 1
        lang_correct[lang] += int(ok)
        if res.source in _CACHE_SOURCES:
            hits += 1
            false_hits += int(not ok)
        oos_tp += int(gold == fallback and res.intent == fallback)
        oos_fp += int(gold != fallback and res.intent == fallback)
        oos_fn += int(gold == fallback and res.intent != fallback)

    def _ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    return {
        "total": total,
        "accuracy": _ratio(correct, total),
        "macro_f1": _macro_f1(y_true, y_pred),
        "hit_rate": _ratio(hits, total),
        "false_hit_rate": _ratio(false_hits, hits),
        "by_source": dict(by_source),
        "latency_ms": {
            s: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
            for s, v in latency.items()
        },
        "per_language_accuracy": {
            lang: _ratio(lang_correct[lang], lang_tot[lang]) for lang in lang_tot
        },
        "oos": {
            "precision": _ratio(oos_tp, oos_tp + oos_fp),
            "recall": _ratio(oos_tp, oos_tp + oos_fn),
        },
    }


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    from sklearn.metrics import f1_score

    labels = sorted(set(y_true) | set(y_pred))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def main() -> None:
    from scripts.seed_cache import load_seed_rows
    from src.embeddings.jina_client import JinaClient
    from src.pipeline import Pipeline

    cfg = load_config()
    rows = load_seed_rows(repo_path(cfg["paths"]["seed_csv"]))
    pipeline = Pipeline(cfg, jina=JinaClient(cfg))
    metrics = evaluate_pipeline(cfg, pipeline, rows)
    import json

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
