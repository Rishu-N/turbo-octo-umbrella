"""End-to-end decision flow (the hot path, §1).  [Phases 3-4: query-only + history escalation]

Orchestrates the layers:
    1. normalize(query)                                  (preprocess/normalize.py)        [Phase 1]
    2. exact-match store (query-only)   -> hit? return    (store/exact_store.py)           [sub-ms]
    3. semantic NN (query alone, T_high) -> hit? return   (jina + store/vector_store.py)
    4. escalate with previous_intent:                                                      [Phase 4]
         a. exact store (query, prev_intent) -> hit? return
         b. semantic NN namespaced by prev_intent, adaptive T_low (optional window fusion) -> hit?
    5. (optional) learned head          -> combiner decision (classifier/combiner.py)      [Phase 5]
    6. LLM fallback (miss)              -> classify        (llm/fallback.py)
    7. OOS gate + confidence-gated write-back (namespaced for referential queries) (§6, §8)
    8. return Classification{intent, confidence, source, is_oos}

``source`` is one of: exact_cache | semantic_cache | head | llm_fallback.
If the Jina API is down, embedding raises ``JinaAPIError`` -> skip the semantic layers, fall through to
the LLM, and **skip write-back** (§2). Step 5 (head) is stubbed until Phase 5.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.classifier.combiner import Combiner
from src.classifier.head import HeadBundle, load_head, predict_proba as head_predict_proba
from src.config import repo_path
from src.embeddings.jina_client import JinaAPIError, JinaClient
from src.llm.fallback import LLMClassifier, StubLLMClassifier
from src.preprocess.normalize import normalize
from src.store.exact_store import ExactStore
from src.store.vector_store import VectorStore

Source = Literal["exact_cache", "semantic_cache", "head", "llm_fallback"]


@dataclass
class Classification:
    """The pipeline's final answer for one request."""

    intent: str
    confidence: float
    source: Source
    is_oos: bool


class Pipeline:
    """Wires normalization, the two cache layers (query-only + escalation), and the LLM fallback.

    Dependencies are injectable (jina/exact/vector/llm) so tests run fully offline; by default they are
    constructed from ``cfg``.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        jina: Any | None = None,
        exact: ExactStore | None = None,
        vector: VectorStore | None = None,
        llm: LLMClassifier | None = None,
        head: HeadBundle | None = None,
    ) -> None:
        self._cfg = cfg
        self._jina = jina if jina is not None else JinaClient(cfg)
        self._exact = exact if exact is not None else ExactStore(cfg)
        self._vector = vector if vector is not None else VectorStore(cfg)
        self._llm = llm if llm is not None else StubLLMClassifier(cfg)

        # Optional learned head (Phase 5). Injected, or loaded from combiner.head_path if present.
        self._combiner_mode = cfg["combiner"]["mode"]
        self._head = head
        if self._head is None and self._combiner_mode != "nn_only":
            head_path = repo_path(cfg["combiner"]["head_path"])
            if head_path.exists():
                self._head = load_head(str(head_path))
        self._combiner = Combiner(cfg, self._head)

        thr = cfg["thresholds"]
        self._t_high = float(thr["t_high"])
        self._t_low = float(thr["t_low"])
        self._t_low_floor = float(thr["t_low_floor"])
        self._t_write = float(thr["t_write"])
        self._oos_floor = float(cfg["llm_fallback"]["oos_confidence_floor"])
        self._fallback_intent = cfg["oos"]["fallback_intent"]
        self._audit_path = repo_path(cfg["audit"]["log_path"])

        hist = cfg["history"]
        self._history_mode = hist.get("mode", "prev_intent")
        self._window_turns = int(hist.get("window_turns", 2))
        self._fusion_weight = float(hist.get("fusion_weight", 0.7))
        self._referential_max = int(hist.get("referential_max_tokens", 4))

    def classify(
        self,
        query: str,
        history: list[str] | None = None,
        previous_intent: str | None = None,
    ) -> Classification:
        """Run the §1 decision flow (query-only, then previous-intent escalation) and return the result."""
        nq = normalize(query, self._cfg)

        # 1) Exact-match, query-only namespace (sub-ms, no embedding).
        hit = self._exact.get(nq)
        if hit is not None:
            return self._from_cache(hit.intent, hit.confidence, "exact_cache")

        # 2) Semantic, query-only namespace (T_high). Embed once; reuse for escalation + write-back.
        query_vec = None
        qo_neighbors: list = []
        try:
            query_vec = self._jina.embed_one(nq)
            qo_neighbors = self._vector.query(query_vec, k=1)  # prev_intent=None -> query-only namespace
            if qo_neighbors and qo_neighbors[0].score >= self._t_high:
                top = qo_neighbors[0]
                return Classification(top.intent, float(top.score), "semantic_cache", self._is_oos(top.intent))
        except JinaAPIError:
            query_vec = None  # §2: API down -> skip semantic layers + write-back

        # 3) Escalation: bring in the previous predicted intent (§3).
        escalating = self._history_mode != "off" and bool(previous_intent)
        if escalating:
            hit2 = self._exact.get(nq, prev_intent=previous_intent)
            if hit2 is not None:
                return self._from_cache(hit2.intent, hit2.confidence, "exact_cache")
            if query_vec is not None:
                fused, frac = self._fuse(query_vec, history)
                eff_t_low = self._effective_t_low(frac)
                neigh2 = self._vector.query(fused, k=5, prev_intent=previous_intent)
                if neigh2 and neigh2[0].score >= eff_t_low:
                    top = neigh2[0]
                    return Classification(top.intent, float(top.score), "semantic_cache", self._is_oos(top.intent))

        # 4) Learned-head stage (combiner cascade). Runs only when the cache missed and a head exists.
        if self._head is not None and query_vec is not None and self._combiner_mode != "nn_only":
            proba = head_predict_proba(self._head, query_vec.reshape(1, -1))[0]
            head_proba = {label: float(p) for label, p in zip(self._head.labels, proba)}
            decision = self._combiner.decide(qo_neighbors, head_proba)
            if decision is not None:
                return Classification(decision.intent, decision.confidence, "head", decision.is_oos)

        # 5) LLM fallback (miss).
        result = self._llm.classify(query, history)
        intent, conf = result.intent, float(result.confidence)

        # 6) Out-of-scope gate (§8): even the LLM is unsure -> fallback. Never written back.
        if conf < self._oos_floor:
            return Classification(self._fallback_intent, conf, "llm_fallback", True)

        # 7) Confidence-gated, audited write-back (§6). Referential/short queries are namespaced by
        #    previous_intent so context-dependent answers don't cross-contaminate (§3).
        if conf >= self._t_write and intent != self._fallback_intent and query_vec is not None:
            wb_prev = previous_intent if (escalating and self._is_referential(nq)) else None
            self._write_back(nq, intent, conf, query_vec, wb_prev)

        return Classification(intent, conf, "llm_fallback", self._is_oos(intent))

    # --- escalation helpers ----------------------------------------------------------------------
    def _effective_t_low(self, frac: float) -> float:
        """Adaptive threshold: more history folded in -> lower threshold, floored (§3)."""
        eff = self._t_low - frac * (self._t_low - self._t_low_floor)
        return max(self._t_low_floor, eff)

    def _fuse(self, query_vec: np.ndarray, history: list[str] | None) -> tuple[np.ndarray, float]:
        """Late-fuse the query with a short history-window embedding (prev_intent_window mode only).

        Returns (vector, frac) where frac in [0, 1] is how much history was folded in (drives the
        adaptive threshold). Falls back to (query_vec, 0.0) when disabled, empty, or on API failure.
        """
        if self._history_mode != "prev_intent_window" or not history:
            return query_vec, 0.0
        window = self._build_window(history)
        if not window:
            return query_vec, 0.0
        try:
            window_vec = self._jina.embed_one(window)
        except JinaAPIError:
            return query_vec, 0.0
        w = self._fusion_weight
        fused = w * query_vec + (1.0 - w) * window_vec
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        frac = min(len(history), self._window_turns) / max(self._window_turns, 1)
        return fused.astype(np.float32), frac

    def _build_window(self, history: list[str]) -> str:
        """Normalize the last ``window_turns`` turns and join with ' [SEP] '."""
        turns = [normalize(t, self._cfg) for t in history[-self._window_turns :]]
        return " [SEP] ".join(t for t in turns if t)

    def _is_referential(self, nq: str) -> bool:
        return len(nq.split()) <= self._referential_max

    # --- internals -------------------------------------------------------------------------------
    def _from_cache(self, intent: str, confidence: float | None, source: Source) -> Classification:
        conf = confidence if confidence is not None else 1.0
        return Classification(intent, conf, source, self._is_oos(intent))

    def _is_oos(self, intent: str) -> bool:
        return intent == self._fallback_intent

    def _write_back(
        self, nq: str, intent: str, conf: float, query_vec: Any, prev_intent: str | None
    ) -> None:
        """Write a high-confidence LLM result into both stores (de-duped, namespaced) and audit it (§6)."""
        result = self._exact.put(nq, intent, "writeback", confidence=conf, prev_intent=prev_intent)
        if result.written:
            self._vector.add(query_vec, intent, "writeback", prev_intent=prev_intent)
            self._vector.save()
        self._audit(
            {
                "ts": time.time(),
                "query": nq,
                "prev_intent": prev_intent or "",
                "intent": intent,
                "confidence": conf,
                "source": "writeback",
                "written": result.written,
                "conflict": result.conflict,
            }
        )

    def _audit(self, record: dict[str, Any]) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
