# Semantic-Cache Intent Classifier (v2)

A **self-improving semantic cache** for multilingual banking intent classification. A new query is
answered by looking it up against a *growing cache* of already-labeled queries instead of always
calling an expensive LLM — and every LLM miss is (safely) written back, so the cache gets smarter
over time **without retraining anything**.

> **Design in one line:** exact-match (local, sub-ms) → semantic NN over Jina v3 embeddings → LLM
> fallback on a miss → confidence-gated write-back that grows the cache.

For the full design, module map, config reference, and data contract, see
**[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** (the handoff doc for any coding agent continuing this
work). Hard constraints live in **[CLAUDE.md](CLAUDE.md)**.

## Decision flow

```
query
  │  normalize  (typed placeholders <AMOUNT>/<CARD>/<PHONE>…, multilingual EN/HI/Hinglish)
  ▼
exact-match store ───hit──▶ intent                         ← sub-millisecond, local, no API call
  │ miss
  ▼
Jina v3 embed (classification adapter, dim 256, L2-norm)  ← one remote API round-trip (tens of ms+)
  │  + vector nearest-neighbor
  ├─ similarity ≥ T_high ─────────────▶ intent             (query alone)
  │  else: bring in previous-intent context (§3)
  ├─ similarity ≥ T_low (adaptive) ───▶ intent
  │  else
  ▼
LLM fallback ───────────────────────▶ intent (+ confidence)   ← swappable interface (stub for now)
  │
  └─ if confidence ≥ T_write: write (normalized query → intent) back into the exact + vector stores
     (de-duplicated, audited, TTL-capped — the self-improvement loop)
```

## Honest latency

Only the **exact-match layer is sub-millisecond** (local lookup). A **semantic-layer hit costs one
Jina API round-trip** (tens of ms or more); an **LLM miss costs more**. We never claim sub-10 ms for
the embedding path — see the note in [`src/embeddings/jina_client.py`](src/embeddings/jina_client.py).

## Status

**All phases (0–6) complete.** The full pipeline is implemented and tested offline (128 tests, ~3s; the
FastAPI test skips without `fastapi`). What's left is operational — run on the work laptop with
`JINA_API_KEY` to seed real data, derive thresholds, and evaluate (see
[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) → *Data contract*). Built phase by phase:

| Phase | What | Status |
|---|---|---|
| 0 | Scaffolding: structure, config, intents, synthetic seed set, stubs, tests | ✅ |
| 1 | `normalize.py`: typed-placeholder multilingual normalization (fully tested) | ✅ |
| 2 | Jina client (mocked in tests) + exact & vector stores | ✅ |
| 3 | Query-only pipeline: exact → semantic(T_high) → LLM stub miss → safe write-back | ✅ |
| 4 | History escalation: `(query, previous_intent)` + adaptive T_low | ✅ |
| 5 | Learned head (LR / nearest-centroid) + NN-then-head combiner + calibration + `fallback` | ✅ |
| 6 | Calibrate thresholds, simulate growth (noisy oracle), audit, eval; optional FastAPI | ✅ |

## Setup

Python 3.10+.

```bash
cd v2
python -m pip install -r requirements.txt          # full deps
export JINA_API_KEY="…"                             # required for the semantic layer (Phase 2+)
```

`JINA_API_KEY` is read from the environment only — never hardcode it. The optional packages in
`requirements.txt` (hnswlib, indic-transliteration, fastapi) are only needed when you enable the
corresponding config feature.

## Run

```bash
cd v2
pip install -r requirements.txt
pytest -q                                  # full suite (128 tests, ~3s)
```

End-to-end workflow on the work laptop (needs `JINA_API_KEY`):

```bash
export JINA_API_KEY=...
python scripts/seed_cache.py               # normalize + embed seed data -> stores (+ train head)
python scripts/calibrate_thresholds.py     # derive T_high / T_low -> copy into config.thresholds
python scripts/simulate_growth.py          # pick T_write from the error-amplification tradeoff
python scripts/evaluate.py                 # accuracy, macro-F1, hit-rate, false-hit, per-lang, latency
python scripts/audit_cache.py              # review cache growth (--purge to roll back write-backs)
uvicorn src.serve:app --port 8080          # optional POST /classify (after `app = create_app()`)
```

`simulate_growth.py` also runs offline (MockEmbedder) to demo the self-improvement / error-amplification
tradeoff without the API.

## Data

Real labeled banking data lives on **your work laptop**, not in this repo. `data/seed/seed.csv` is a
tiny **synthetic** multilingual placeholder (EN / Hindi / Hinglish, confusable intents, multi-entity
examples) for development and tests. All data paths are config-driven; drop your real CSV in
`data/raw/` (or point `paths.seed_csv` at it) and follow the **Data contract** section of
[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md). Adding an intent never retrains a base model — see
[`data/intents.yaml`](data/intents.yaml).
