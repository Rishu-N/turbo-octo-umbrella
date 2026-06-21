"""Exact-match store — sub-millisecond local key -> intent lookup + metadata.  [Phase 2]

The first cache layer (§1 step 2): if the normalized query (namespaced by previous intent on the
escalation path, §3) is already present, return its intent immediately with NO embedding call. This is
the ONLY sub-millisecond path. Backed by SQLite (config: exact_store.path), which also carries the
write-back metadata the safety layer needs (§6): timestamp, source (seed vs writeback), confidence, and
dedup/conflict records.

Write-back safety semantics (§6):
    - same key, same intent           -> no-op (refresh last_used_at).
    - same key, different intent      -> CONFLICT. Recorded in a `conflicts` table (audit.record_conflicts);
                                         seed entries are NEVER overwritten; a writeback never flip-flops
                                         an existing intent; only a `seed` put overrides a prior `writeback`.
    - prev_intent namespaces the key  -> ("yes", paying_bill) and ("yes", transfer) are distinct buckets.

Public surface:
    ExactStore(cfg)
      .get(key, prev_intent=None) -> CacheEntry | None
      .put(key, intent, source, confidence=None, prev_intent=None) -> PutResult
      .evict(now=None) -> int        # apply TTL / size caps (seed entries are protected)
      .stats() -> dict
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Literal

from src.config import repo_path

Source = Literal["seed", "writeback"]


@dataclass
class CacheEntry:
    """A single exact-store record."""

    key: str                  # normalized query (the lookup key)
    intent: str
    source: Source            # "seed" (ground truth, never evicted) | "writeback" (LLM-derived)
    confidence: float | None  # None for seed; the LLM confidence for write-back
    prev_intent: str | None   # namespacing for the escalation path (§3); None on the query-only path
    created_at: float         # unix epoch seconds
    last_used_at: float       # unix epoch seconds (for LRU eviction)


@dataclass
class PutResult:
    """Outcome of a put(): whether it was written, and any dedup conflict (§6)."""

    written: bool
    conflict: bool            # True if key already mapped to a DIFFERENT intent
    existing_intent: str | None


def _ns(prev_intent: str | None) -> str:
    """Namespace token: None (query-only path) is stored as '' to keep the composite PK well-defined."""
    return prev_intent or ""


class ExactStore:
    """SQLite-backed exact key -> intent store with write-back metadata + eviction."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        path = repo_path(cfg["exact_store"]["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  key TEXT NOT NULL, prev_intent TEXT NOT NULL, intent TEXT NOT NULL,"
            "  source TEXT NOT NULL, confidence REAL,"
            "  created_at REAL NOT NULL, last_used_at REAL NOT NULL,"
            "  PRIMARY KEY (key, prev_intent))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS conflicts ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, prev_intent TEXT,"
            "  existing_intent TEXT, attempted_intent TEXT, attempted_source TEXT, created_at REAL)"
        )
        self._conn.commit()

    # --- reads -----------------------------------------------------------------------------------
    def get(self, key: str, prev_intent: str | None = None) -> CacheEntry | None:
        """Return the entry for ``key`` (optionally namespaced by ``prev_intent``), or None."""
        ns = _ns(prev_intent)
        row = self._conn.execute(
            "SELECT key, prev_intent, intent, source, confidence, created_at, last_used_at "
            "FROM entries WHERE key = ? AND prev_intent = ?",
            (key, ns),
        ).fetchone()
        if row is None:
            return None
        now = time.time()
        self._conn.execute(
            "UPDATE entries SET last_used_at = ? WHERE key = ? AND prev_intent = ?", (now, key, ns)
        )
        self._conn.commit()
        return CacheEntry(
            key=row[0], prev_intent=row[1] or None, intent=row[2], source=row[3],
            confidence=row[4], created_at=row[5], last_used_at=now,
        )

    # --- writes ----------------------------------------------------------------------------------
    def put(
        self,
        key: str,
        intent: str,
        source: Source,
        confidence: float | None = None,
        prev_intent: str | None = None,
    ) -> PutResult:
        """Insert/refresh an entry, applying the write-back safety semantics (see module docstring)."""
        ns = _ns(prev_intent)
        now = time.time()
        row = self._conn.execute(
            "SELECT intent, source FROM entries WHERE key = ? AND prev_intent = ?", (key, ns)
        ).fetchone()

        if row is None:
            self._conn.execute(
                "INSERT INTO entries(key, prev_intent, intent, source, confidence, created_at, last_used_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, ns, intent, source, confidence, now, now),
            )
            self._conn.commit()
            return PutResult(written=True, conflict=False, existing_intent=None)

        existing_intent, existing_source = row
        if existing_intent == intent:
            self._conn.execute(
                "UPDATE entries SET last_used_at = ? WHERE key = ? AND prev_intent = ?", (now, key, ns)
            )
            self._conn.commit()
            return PutResult(written=False, conflict=False, existing_intent=existing_intent)

        # Different intent for the same key -> conflict. Record it; do not flip-flop.
        self._record_conflict(key, ns, existing_intent, intent, source, now)
        if source == "seed" and existing_source != "seed":
            # Ground truth overrides a prior write-back.
            self._conn.execute(
                "UPDATE entries SET intent = ?, source = ?, confidence = ?, created_at = ?, last_used_at = ?"
                " WHERE key = ? AND prev_intent = ?",
                (intent, source, confidence, now, now, key, ns),
            )
            self._conn.commit()
            return PutResult(written=True, conflict=True, existing_intent=existing_intent)
        return PutResult(written=False, conflict=True, existing_intent=existing_intent)

    def _record_conflict(
        self, key: str, ns: str, existing_intent: str, attempted_intent: str, source: str, now: float
    ) -> None:
        if not self._cfg.get("audit", {}).get("record_conflicts", True):
            return
        self._conn.execute(
            "INSERT INTO conflicts(key, prev_intent, existing_intent, attempted_intent, attempted_source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (key, ns, existing_intent, attempted_intent, source, now),
        )
        self._conn.commit()

    # --- maintenance -----------------------------------------------------------------------------
    def evict(self, now: float | None = None) -> int:
        """Apply TTL (eviction.ttl_days) and size cap (eviction.max_entries). Seed entries protected."""
        now = now if now is not None else time.time()
        ev = self._cfg["eviction"]
        protect_seed = bool(ev.get("protect_seed", True))
        where_evictable = "source = 'writeback'" if protect_seed else "1 = 1"
        deleted = 0

        ttl_days = ev.get("ttl_days", 0)
        if ttl_days and ttl_days > 0:
            cutoff = now - ttl_days * 86400
            cur = self._conn.execute(
                f"DELETE FROM entries WHERE {where_evictable} AND created_at < ?", (cutoff,)
            )
            deleted += cur.rowcount
            self._conn.commit()

        max_entries = ev.get("max_entries", 0)
        if max_entries and max_entries > 0:
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM entries WHERE {where_evictable}"
            ).fetchone()[0]
            if count > max_entries:
                excess = count - max_entries
                rows = self._conn.execute(
                    f"SELECT rowid FROM entries WHERE {where_evictable} ORDER BY last_used_at ASC LIMIT ?",
                    (excess,),
                ).fetchall()
                self._conn.executemany(
                    "DELETE FROM entries WHERE rowid = ?", [(r[0],) for r in rows]
                )
                deleted += len(rows)
                self._conn.commit()
        return deleted

    def stats(self) -> dict[str, Any]:
        """Return counts by source/intent and the conflict tally (for scripts/audit_cache.py)."""
        c = self._conn
        total = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        by_source = dict(c.execute("SELECT source, COUNT(*) FROM entries GROUP BY source").fetchall())
        per_intent = dict(c.execute("SELECT intent, COUNT(*) FROM entries GROUP BY intent").fetchall())
        conflicts = c.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
        return {"total": total, "by_source": by_source, "per_intent": per_intent, "conflicts": conflicts}

    def conflicts(self) -> list[dict[str, Any]]:
        """Return recorded write-back conflicts (key, namespace, existing vs attempted intent)."""
        rows = self._conn.execute(
            "SELECT key, prev_intent, existing_intent, attempted_intent, attempted_source, created_at "
            "FROM conflicts ORDER BY created_at"
        ).fetchall()
        keys = ("key", "prev_intent", "existing_intent", "attempted_intent", "attempted_source", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def purge_writebacks(self, intent: str | None = None) -> int:
        """Delete write-back entries (optionally only for one intent); seed entries are untouched."""
        sql = "DELETE FROM entries WHERE source = 'writeback'"
        params: tuple = ()
        if intent is not None:
            sql += " AND intent = ?"
            params = (intent,)
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount
