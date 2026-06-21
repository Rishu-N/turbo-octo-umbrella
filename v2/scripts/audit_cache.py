"""Audit the cache: conflicts, write-back entries, per-intent counts, and purge.  [Phase 6]

Reviews cache growth (from the exact store + its conflict log) so bad write-back entries can be found and
removed. Seed (ground-truth) entries are never purged.

Usage:
    python scripts/audit_cache.py                      # print summary + conflicts
    python scripts/audit_cache.py --purge              # delete ALL write-back entries
    python scripts/audit_cache.py --purge --intent X   # delete write-back entries for intent X
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402


def summarize(exact: Any) -> dict:
    """Return store stats plus the recorded write-back conflicts."""
    stats = exact.stats()
    stats["conflict_details"] = exact.conflicts()
    return stats


def _print_summary(summary: dict) -> None:
    print(f"total entries     : {summary['total']}")
    print(f"by source         : {summary['by_source']}")
    print(f"conflicts recorded: {summary['conflicts']}")
    print("per-intent counts :")
    for intent, count in sorted(summary["per_intent"].items(), key=lambda kv: -kv[1]):
        print(f"    {intent:<26} {count}")
    if summary["conflict_details"]:
        print("conflicts:")
        for c in summary["conflict_details"]:
            ns = c["prev_intent"] or "(query-only)"
            print(f"    key={c['key']!r} [{ns}] {c['existing_intent']} <- attempted {c['attempted_intent']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit / purge the write-back cache.")
    parser.add_argument("--purge", action="store_true", help="delete write-back entries")
    parser.add_argument("--intent", default=None, help="restrict --purge to a single intent")
    args = parser.parse_args()

    from src.store.exact_store import ExactStore

    cfg = load_config()
    exact = ExactStore(cfg)
    _print_summary(summarize(exact))
    if args.purge:
        removed = exact.purge_writebacks(intent=args.intent)
        print(f"\npurged {removed} write-back entries" + (f" for intent={args.intent}" if args.intent else ""))


if __name__ == "__main__":
    main()
