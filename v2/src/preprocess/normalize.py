"""Typed-placeholder text normalization (multilingual: EN / Hindi / Hinglish).  [Phase 1]

Normalizes a raw query into a canonical form BEFORE exact-match keying and embedding, so that
different specifics ("$5,000", "card 4111 1111 1111 1111") collapse to the same canonical string and
lift the cache hit-rate. Uses *typed* placeholders (``<AMOUNT>``, ``<CARD>``, ``<PHONE>`` …) rather than
one generic token, because the entity type preserves intent signal.

Banking caveat (§4): sometimes the number IS the signal ("charged twice", "$0.00", a 16-digit card vs a
10-digit phone). So entities are distinguished by **length/pattern**, and a configurable allowlist of
meaningful tokens (config: ``normalization.meaningful_tokens_allowlist``) is preserved verbatim.

Pipeline order (each step is config-gated; order matters and is deliberate):
  1. NFC unicode normalize + whitespace collapse.
  2. Devanagari numerals (०-९) -> ASCII, so digit/length rules work on Hindi numerals too.
  3. (optional) Romanized Hindi -> Devanagari transliteration (off by default; adds latency).
  4. lowercase (so placeholder tags inserted afterwards stay UPPERCASE).
  5. Typed-placeholder replacement, longest/most-specific first:
     URL -> EMAIL -> AMOUNT (currency-tagged) -> DATE -> CARD/PHONE (by digit count) -> MONTH ->
     ID (alnum w/ a digit run) -> NUM (leftover digits). Allowlisted numeric tokens are left verbatim.
  6. (optional) strip punctuation — decimal-preserving, so "0.00" survives but "balance?" -> "balance".

Public surface:
    normalize(text, cfg) -> str        # the canonical, cache-keyable string

NOTE on transliteration: free-form Hinglish -> Devanagari really needs a model (e.g. AI4Bharat
IndicXlit). The optional path here is a lightweight ``indic-transliteration`` (ITRANS) fallback and is
imperfect; keep it off until a proper engine is wired in.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# --- defaults (used when a config key is absent) -------------------------------------------------
_DEFAULT_ALLOW = {"0", "0.0", "0.00", "zero", "twice", "double"}

# --- character tables / small regexes ------------------------------------------------------------
# Devanagari digit ० (U+0966) .. ९ (U+096F) -> ASCII 0..9
_DEVANAGARI_DIGITS = {0x0966 + i: str(i) for i in range(10)}
_WS = re.compile(r"\s+")
_APOSTROPHES = re.compile(r"['‘’`´]")
_TAG = re.compile(r"<[A-Z]+>")  # an already-inserted placeholder token, e.g. <AMOUNT>
_DOT_NOT_AFTER_DIGIT = re.compile(r"(?<!\d)\.")
_DOT_NOT_BEFORE_DIGIT = re.compile(r"\.(?!\d)")

# --- entity patterns (compiled once; enabling is decided per-call from config) -------------------
_URL = re.compile(r"(?<![\w])(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_EMAIL = re.compile(r"(?<![\w])[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}(?![\w])", re.IGNORECASE)

# Currency-tagged amounts only (a bare number is handled by <NUM>, not <AMOUNT>).
_AMOUNT = re.compile(
    r"""
      (?:
         [$₹€£¥]\s?\d[\d,]*(?:\.\d+)?                                            # $500 ; ₹1,000.50
       | (?:rs\.?|inr|usd|eur|gbp)\s?\d[\d,]*(?:\.\d+)?                          # rs 500 ; inr500
       | \d[\d,]*(?:\.\d+)?\s?(?:rs\.?|rupees?|inr|usd|dollars?|eur|euros?|gbp|pounds?|[$₹€£¥])  # 500 rupees ; 500$
      )
    """,
    re.IGNORECASE | re.VERBOSE | re.UNICODE,
)

# Numeric dates: 12/05/2024, 2024-05-12, 12-5-24 (middle group <= 2 digits, so 4-group cards don't match).
_DATE = re.compile(r"(?<![\w])\d{1,4}[/-]\d{1,2}[/-]\d{2,4}(?![\w])")

# A run of 9-19 digits with optional single space/hyphen separators and a leading '+': classified by
# digit count into CARD (13-19) or PHONE (10-12). Bounded so amounts/dates (already replaced) and short
# numbers are left to <NUM>.
_DIGIT_RUN = re.compile(r"(?<![\w/.,])\+?\d(?:[ \-]?\d){8,18}(?![\w/.,])")

# Month names: English (full + abbrev) excluding the ambiguous "may"; plus common Devanagari spellings
# (Devanagari "मई" is unambiguous, so it is included). Longest-first so "september" beats "sep".
_MONTH_WORDS = sorted(
    [
        "january", "february", "march", "april", "june", "july", "august",
        "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "जनवरी", "फरवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त",
        "सितंबर", "सितम्बर", "अक्टूबर", "अक्तूबर", "नवंबर", "नवम्बर", "दिसंबर", "दिसम्बर",
    ],
    key=len,
    reverse=True,
)
_MONTH = re.compile(
    r"(?<![\w])(?:" + "|".join(re.escape(w) for w in _MONTH_WORDS) + r")(?![\w])",
    re.IGNORECASE | re.UNICODE,
)

# Reference/transaction ids: a token with >=1 letter AND a run of >=3 digits, length >= 6
# (e.g. "txn12345", "ref998877"). Conservative, to avoid catching words like "covid19".
_ID = re.compile(r"(?<![\w])(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d{3})[a-z0-9]{6,}(?![\w])", re.IGNORECASE)

# Leftover bare numbers (after the rules above): "199", "5,000", "0.00".
_NUM = re.compile(r"(?<![\w])\d[\d,]*(?:\.\d+)?(?![\w])")


def _maybe_transliterate(text: str, enabled: bool) -> str:
    """Optionally transliterate Romanized Hindi -> Devanagari (config-gated; off by default)."""
    if not enabled:
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError as e:  # pragma: no cover - exercised only when the feature is enabled
        raise ImportError(
            "normalization.transliterate_romanized=True requires `pip install indic-transliteration`"
        ) from e
    # Best-effort ITRANS->Devanagari. Imperfect for free-form Hinglish; see module docstring.
    return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)


def _amount_sub(m: re.Match[str], allow: set[str]) -> str:
    """Replace a currency-tagged amount with <AMOUNT>, unless its value is allowlisted (e.g. $0.00)."""
    core = re.sub(r"[^\d.]", "", m.group(0))
    return m.group(0) if core in allow else "<AMOUNT>"


def _num_sub(m: re.Match[str], allow: set[str]) -> str:
    """Replace a bare number with <NUM>, unless it is an allowlisted meaningful token (e.g. 0, 0.00)."""
    core = m.group(0).replace(",", "")
    return m.group(0) if core in allow else "<NUM>"


def _digit_run_sub(m: re.Match[str], card_on: bool, phone_on: bool) -> str:
    """Classify a digit run as <CARD> (13-19 digits) or <PHONE> (10-12); otherwise leave it."""
    n = len(re.sub(r"\D", "", m.group(0)))
    if card_on and 13 <= n <= 19:
        return "<CARD>"
    if phone_on and 10 <= n <= 12:
        return "<PHONE>"
    return m.group(0)


def _lower_preserving_tags(s: str) -> str:
    """Lowercase everything except already-inserted <TAG> tokens (keeps normalize() idempotent)."""
    parts: list[str] = []
    last = 0
    for m in _TAG.finditer(s):
        parts.append(s[last:m.start()].lower())
        parts.append(m.group(0))
        last = m.end()
    parts.append(s[last:].lower())
    return "".join(parts)


def _strip_punctuation(s: str) -> str:
    """Remove punctuation/symbols but keep placeholder tags and decimals (e.g. '0.00').

    Uses Unicode *categories* (keep Letters/Marks/Numbers) rather than ``\\w`` — crucially, ``\\w``
    does NOT match Devanagari combining vowel marks (matras), so a ``\\w``-based strip shreds Hindi.
    """
    s = _APOSTROPHES.sub("", s)  # contractions: what's -> whats
    kept = [
        ch if (ch.isspace() or ch in "<>._" or unicodedata.category(ch)[0] in ("L", "M", "N")) else " "
        for ch in s
    ]
    s = "".join(kept)
    s = _DOT_NOT_AFTER_DIGIT.sub(" ", s)   # keep dots only between digits ("0.00"); drop "balance."
    s = _DOT_NOT_BEFORE_DIGIT.sub(" ", s)
    return s


def normalize(text: str | None, cfg: dict[str, Any] | None) -> str:
    """Return the canonical normalized form of ``text`` per the config ``normalization`` settings.

    Args:
        text: the raw query (``None``/empty -> ``""``).
        cfg: the full config dict; only ``cfg["normalization"]`` is read (missing keys fall back to
            sensible defaults, so callers can pass a partial/empty config).

    Returns:
        The normalized, cache-keyable string.
    """
    if not text:
        return ""

    norm = (cfg or {}).get("normalization", {}) or {}
    placeholders = norm.get("placeholders", {}) or {}

    def on(kind: str) -> bool:
        return bool(placeholders.get(kind, True))

    allow = {
        str(t).strip().lower().replace(",", "")
        for t in norm.get("meaningful_tokens_allowlist", _DEFAULT_ALLOW)
    }

    s = unicodedata.normalize("NFC", str(text))
    s = _WS.sub(" ", s).strip()

    if norm.get("devanagari_numerals", True):
        s = s.translate(_DEVANAGARI_DIGITS)
    s = _maybe_transliterate(s, norm.get("transliterate_romanized", False))
    if norm.get("lowercase", True):
        s = _lower_preserving_tags(s)

    if on("url"):
        s = _URL.sub("<URL>", s)
    if on("email"):
        s = _EMAIL.sub("<EMAIL>", s)
    if on("amount"):
        s = _AMOUNT.sub(lambda m: _amount_sub(m, allow), s)
    if on("date"):
        s = _DATE.sub("<DATE>", s)
    if on("card") or on("phone"):
        s = _DIGIT_RUN.sub(lambda m: _digit_run_sub(m, on("card"), on("phone")), s)
    if on("month"):
        s = _MONTH.sub("<MONTH>", s)
    if on("id"):
        s = _ID.sub("<ID>", s)
    if on("num"):
        s = _NUM.sub(lambda m: _num_sub(m, allow), s)

    if norm.get("strip_punctuation", True):
        s = _strip_punctuation(s)

    return _WS.sub(" ", s).strip()
