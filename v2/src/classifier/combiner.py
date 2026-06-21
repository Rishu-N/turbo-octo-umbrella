"""Combiner — how the NN cache and the learned head produce one decision.  [Phase 5]

Explicit, configurable combination (config: combiner.mode): nn_only | head_only | nn_then_head. This is
the **learned-head stage** of the pipeline — it runs after the cache layers (exact + semantic NN) have
missed their thresholds and before the LLM fallback. It returns a `Decision` to accept, or `None` to
defer to the LLM (which, with the OOS gate, yields the `fallback` outcome, §8).

    - nn_only:      head disabled -> always `None` (the NN cache is the only learned component).
    - head_only:    accept the head's top intent if its calibrated prob >= combiner.head_threshold.
    - nn_then_head: like head_only, but if the best (sub-threshold) NN neighbor agrees with the head's
                    top intent, reinforce confidence to max(head_prob, nn_score) — the cascade.

Public surface:
    Combiner(cfg, head=None).decide(neighbors, head_proba=None) -> Decision | None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.store.vector_store import Neighbor


@dataclass
class Decision:
    """The combiner's verdict for one query (the pipeline maps source -> 'head')."""

    intent: str
    confidence: float
    source: str        # "head"
    is_oos: bool


class Combiner:
    """Implements the NN / head / cascade decision logic over calibrated head probabilities."""

    def __init__(self, cfg: dict[str, Any], head: Any | None = None) -> None:
        self._cfg = cfg
        self._head = head
        c = cfg["combiner"]
        self._mode = c["mode"]
        self._head_threshold = float(c.get("head_threshold", 0.5))
        self._fallback_intent = cfg.get("oos", {}).get("fallback_intent", "fallback")

    def decide(
        self,
        neighbors: "list[Neighbor]",
        head_proba: dict[str, float] | None = None,
    ) -> Decision | None:
        """Combine NN neighbors and calibrated head probabilities; return a Decision or None (defer)."""
        if self._mode == "nn_only" or not head_proba:
            return None

        top_intent = max(head_proba, key=head_proba.get)
        confidence = float(head_proba[top_intent])

        # Cascade: a sub-threshold NN neighbor that agrees reinforces the head's confidence.
        if self._mode == "nn_then_head" and neighbors and neighbors[0].intent == top_intent:
            confidence = max(confidence, float(neighbors[0].score))

        if confidence >= self._head_threshold:
            return Decision(top_intent, confidence, "head", top_intent == self._fallback_intent)
        return None
