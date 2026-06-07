"""Build the model input string from (history, current_query) per CLAUDE.md §7.

Layout:  "{query_prefix}{history}{separator}{current_query}"

- query_prefix:  e.g. "query: " (required by intfloat/e5 models)
- separator:     " [SEP] "
- truncation:    leave to the tokenizer (controlled by encoder.max_seq_length)

Optional language-ID and Romanized→Devanagari transliteration are gated by
config flags and stubbed as no-ops until Phase 5.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_input(query: str, history: str, cfg: dict[str, Any]) -> str:
    """Return the single string to feed into the encoder."""
    prefix = cfg["encoder"]["query_prefix"] or ""
    sep = cfg["history"]["separator"]

    query = (query or "").strip()
    history = (history or "").strip()

    if cfg["preprocess"].get("transliterate_romanized", False):
        history = _maybe_transliterate(history)
        query = _maybe_transliterate(query)

    if history:
        return f"{prefix}{history}{sep}{query}"
    return f"{prefix}{query}"


def build_inputs(df: pd.DataFrame, cfg: dict[str, Any], out_col: str = "text") -> pd.DataFrame:
    """Add an `out_col` column with the built input string for every row."""
    df = df.copy()
    df[out_col] = [build_input(q, h, cfg) for q, h in zip(df["query"], df["history"])]
    return df


def _maybe_transliterate(text: str) -> str:
    """Stub: Romanized-Hindi → Devanagari. Wired up in Phase 5."""
    # TODO(Phase 5): use indic-transliteration when preprocess.transliterate_romanized is on.
    return text
