"""LLM fallback — interface + stub (the real classifier plugs in here).  [interface ✅; stub Phase 3]

On a cache miss, the pipeline calls the existing LLM classifier. We depend only on this clean interface
so the real implementation is trivially swappable (§5): subclass ``LLMClassifier``, return an
``LLMResult``, and select it via ``config.llm_fallback.impl``.

For now ``StubLLMClassifier`` stands in so the whole system works end-to-end offline. It resolves an
intent by, in order: (1) an explicit ``mapping`` (raw query -> intent or (intent, confidence)) for
deterministic tests/demos, (2) simple keyword rules over the lowercased query, (3) a low-confidence
``fallback`` for anything unrecognized (which the pipeline turns into an OOS result).

Public surface:
    LLMClassifier (ABC).classify(query, history) -> LLMResult
    StubLLMClassifier(cfg, mapping=None)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResult:
    """What every LLM classifier returns: a predicted intent and a [0, 1] confidence."""

    intent: str
    confidence: float


class LLMClassifier(ABC):
    """Interface for the existing LLM intent classifier. Implementations MUST be swappable."""

    @abstractmethod
    def classify(self, query: str, history: list[str] | None = None) -> LLMResult:
        """Classify ``query`` (with optional ``history`` = prior turn texts) into intent + confidence."""
        raise NotImplementedError


class StubLLMClassifier(LLMClassifier):
    """Offline stand-in for the real classifier: explicit mapping -> keyword rules -> low-conf fallback."""

    # First match wins; ordered so more specific cues precede generic ones (e.g. "twice" before "charge").
    _KEYWORD_RULES: list[tuple[str, str]] = [
        ("balance", "balance_inquiry"),
        ("lost", "card_lost_or_stolen"), ("stolen", "card_lost_or_stolen"),
        ("kho gaya", "card_lost_or_stolen"), ("block", "card_lost_or_stolen"),
        ("declined", "card_payment_declined"), ("decline", "card_payment_declined"),
        ("not working", "card_not_working"), ("kaam nahi", "card_not_working"),
        ("activate", "card_activation"),
        ("twice", "duplicate_charge"), ("double", "duplicate_charge"), ("do baar", "duplicate_charge"),
        ("recognize", "unrecognized_transaction"), ("unknown", "unrecognized_transaction"),
        ("nahi kiya", "unrecognized_transaction"),
        ("pending", "transaction_pending"),
        ("refund", "refund_status"),
        ("limit", "transfer_limit_inquiry"),
        ("transfer", "transfer_funds"), ("send money", "transfer_funds"),
        ("statement", "account_statement"),
        ("pin", "change_pin"),
        ("phone number", "update_contact_details"), ("email", "update_contact_details"),
        ("address", "update_contact_details"),
        ("loan", "loan_inquiry"),
        ("rate", "exchange_rate"),
        ("fee", "fee_explanation"), ("charge", "fee_explanation"),
        ("atm", "atm_issue"),
    ]

    def __init__(self, cfg: dict[str, Any], mapping: dict[str, Any] | None = None) -> None:
        self._cfg = cfg
        lf = cfg.get("llm_fallback", {})
        self._default_confidence = float(lf.get("stub_default_confidence", 0.75))
        self._fallback_intent = cfg.get("oos", {}).get("fallback_intent", "fallback")
        # Raw query -> intent, or raw query -> (intent, confidence). For deterministic tests/demos.
        self._mapping = mapping or {}

    def classify(self, query: str, history: list[str] | None = None) -> LLMResult:
        if query in self._mapping:
            value = self._mapping[query]
            if isinstance(value, (tuple, list)):
                return LLMResult(str(value[0]), float(value[1]))
            return LLMResult(str(value), self._default_confidence)

        q = (query or "").lower()
        for keyword, intent in self._KEYWORD_RULES:
            if keyword in q:
                return LLMResult(intent, self._default_confidence)

        # Unrecognized -> deliberately low confidence so the pipeline returns OOS/fallback (§8).
        return LLMResult(self._fallback_intent, 0.30)
