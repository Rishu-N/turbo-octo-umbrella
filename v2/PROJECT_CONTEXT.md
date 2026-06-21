# PROJECT_CONTEXT.md — handoff & context for the next coding agent

> **Read this before touching the code.** It is the single source of truth for *what* this project is,
> *why* it is built this way, and *how* to extend or change it. It is a **living document**: it is
> updated at the end of every build phase so it always matches the code. Hard rules also live in
> `CLAUDE.md` (which points here); user-facing run instructions live in `README.md`.
>
> **Current state: ALL phases (0–6) complete.** The full system is implemented and tested offline
> (128 tests; the FastAPI test skips if `fastapi` isn't installed): normalization, Jina client, both
> stores, escalation, the learned head + combiner, the pipeline, and the Phase 6 scripts
> (seed_cache / calibrate_thresholds / simulate_growth / audit_cache / evaluate) + optional FastAPI.
> What remains is **operational, not code**: run on the work laptop with `JINA_API_KEY` to seed real
> data, derive thresholds, and evaluate (see *Data contract* and *Phase status*).

---

## 1. What this is, and why

A **self-improving semantic-cache intent classifier** for a multilingual banking chatbot. Instead of
calling an expensive LLM on every turn, a query is classified by **looking it up against a growing
cache of already-labeled queries**. Misses fall through to the LLM, whose answer is **written back**
into the cache behind a confidence gate — so the hit-rate rises over time and **nothing is retrained**.
The "learning" is *cache growth*, not model training.

**The decision flow (the hot path):**
1. **Normalize** the query → typed placeholders, multilingual (`preprocess/normalize.py`).
2. **Exact-match** layer (local, sub-ms): normalized query already in the key→intent store? Return it.
   No embedding call.
3. **Semantic** layer (Jina v3 API): else embed the normalized query and do nearest-neighbor lookup.
   - top similarity ≥ **T_high** → return that intent (query alone).
   - else bring in **conversation history** cheaply (previous predicted intent) and look up again with
     a lower, adaptive **T_low** (with a floor). Clears T_low → return it.
4. **LLM fallback** (miss): nothing cleared threshold → call the existing LLM classifier (behind a
   clean interface; stubbed for now).
5. **Write-back** (self-improvement): after an LLM fallback, write `normalized query → intent` into
   **both** stores — but **only if** the LLM confidence ≥ **T_write** (plus de-dup + audit + TTL).

**Why a cache and not just a model:** new intents are added frequently and must not require retraining;
most banking queries are self-contained and repetitive (high cache potential); and the exact-match
layer gives a genuinely sub-ms path for the common case.

---

## 2. Hard constraints & guardrails (do NOT violate)

These are non-negotiable. They come from the project brief (§13 "things NOT to do") and shape every
module.

- **Jina v3 is API-only.** No local weights. **Every `embed()` is a remote network call.** Treat it as
  remote everywhere: batch, timeout, retry-with-backoff, and **fail gracefully** — if the API is down,
  fall straight through to the LLM fallback and **skip write-back**. Do **not** call it sub-10 ms or
  "local". Only the exact-match layer is sub-ms.
- **Use the Jina `classification` task adapter** (intent-tuned LoRA), **Matryoshka dim 256**, and
  **L2-normalize** so cosine == dot product. All three are config values.
- **`JINA_API_KEY` comes from the environment.** Never hardcode it; never put it in `config.yaml`.
  (Enforced by `tests/test_config.py::test_no_api_key_in_config`.)
- **Do not concatenate full conversation history into the embedded text** — high-entropy history
  destroys the cache hit-rate. The primary key is the **normalized current query alone**; only escalate
  to history (previous intent) when the query fails T_high, and keep previous-intent in the key so the
  same surface query ("yes") in different contexts lands in different buckets.
- **Typed-placeholder normalization, not blanket digit-stripping.** Distinguish card vs phone vs amount
  by length/pattern; keep a configurable allowlist of meaningful tokens ("0", "$0.00", "twice", …) —
  sometimes the number *is* the signal.
- **Write-back safety (error-amplification guard).** Only write back if confidence ≥ T_write; de-dup
  (same key + different intent → record a conflict, don't flip-flop); TTL + LRU/size caps with **seed
  entries never evicted**; and an **audit trail** of every write-back.
- **Thresholds are derived, not guessed.** T_high / T_low / T_write must come from
  `scripts/calibrate_thresholds.py` (Phase 6) on held-out data. The values currently in `config.yaml`
  are **placeholders**; any number from the literature is for a different model and must be re-derived
  for Jina v3.
- **Adding an intent never retrains a base model.** Refit the cheap head / add a centroid (Phase 5) or
  add seed/cache entries — see *How to extend*.
- **No network calls except the isolated Jina client** (and the real LLM later). Everything else is
  local and offline-testable (the Jina client is mocked in tests).

---

## 3. Architecture & module map

The hot path is `src/pipeline.py`, which wires the pieces below. Status: ✅ functional · 🔌 stub
(contract defined, body raises `NotImplementedError`).

| Module | Role | Public surface | Status |
|---|---|---|---|
| `src/config.py` | Load `config.yaml` as a dict; resolve repo-relative paths. | `load_config()`, `repo_path()`, `REPO_ROOT` | ✅ |
| `src/preprocess/normalize.py` | Typed-placeholder multilingual normalization (EN/HI/Hinglish). | `normalize(text, cfg) -> str` | ✅ |
| `src/embeddings/jina_client.py` | Isolated Jina v3 API client: classification adapter, dim 256, L2-norm, retries/timeouts, persistent embed cache keyed by normalized text. | `JinaClient(cfg).embed(texts)`, `.embed_one(text)`; `JinaAPIError` | ✅ |
| `src/store/exact_store.py` | Sub-ms key→intent store (SQLite) + write-back metadata (source, timestamp, confidence, conflicts) + eviction. | `ExactStore(cfg)`: `get/put/evict/stats`; `CacheEntry`, `PutResult` | ✅ |
| `src/store/vector_store.py` | Semantic NN index (hnswlib) over L2-normalized embeddings + id→meta sidecar, with `prev_intent` namespacing. | `VectorStore(cfg)`: `add/query/save/load/size`; `Neighbor` | ✅ |
| `src/classifier/head.py` | Learned head over embeddings (logistic regression / nearest-centroid) + temperature calibration. | `train_head`, `predict_proba`, `calibrate_temperature`, `save_head`, `load_head`; `HeadBundle` | ✅ |
| `src/classifier/combiner.py` | Combine NN cache + head into one decision (nn_only / head_only / cascade); defers to LLM via `None`. | `Combiner(cfg, head).decide(...)`; `Decision` | ✅ |
| `src/llm/fallback.py` | Swappable LLM-classifier interface + offline stub. | `LLMClassifier` (ABC), `StubLLMClassifier`; `LLMResult` | ✅ |
| `src/pipeline.py` | End-to-end §1 decision flow: query-only + previous-intent escalation + gated write-back. | `Pipeline(cfg).classify(query, history, previous_intent)`; `Classification` | ✅ |
| `src/serve.py` | Optional FastAPI `POST /classify` (fastapi imported lazily; inject a pipeline for tests). | `create_app(cfg, pipeline)` | ✅ |
| `src/embeddings/mock.py` | Deterministic offline embedder for dev/scripts/tests (no semantics). | `MockEmbedder(dim)` | ✅ |
| `scripts/seed_cache.py` | Normalize+embed seed data → stores (source="seed", namespaced) + train head. | `seed_stores`, `train_and_save_head`, `load_seed_rows`, `main` | ✅ |
| `scripts/calibrate_thresholds.py` | Sweep → derive T_high/T_low; hit-rate vs false-hit curves. | `collect_scores`, `sweep_thresholds`, `recommend`, `main` | ✅ |
| `scripts/simulate_growth.py` | Replay stream w/ noisy oracle → error-amplification vs T_write. | `simulate`, `sweep_t_write`, `make_stream`, `NoisyOracle`, `main` | ✅ |
| `scripts/audit_cache.py` | List conflicts / write-backs / per-intent counts; purge. | `summarize`, `main` (+ `ExactStore.conflicts/purge_writebacks`) | ✅ |
| `scripts/evaluate.py` | Cache + intent + OOS metrics + latency-per-path (§12). | `evaluate_pipeline`, `main` | ✅ |

`source` values returned by the pipeline: `exact_cache | semantic_cache | head | llm_fallback`.
Entry `source` in the stores: `seed` (ground truth, never evicted) | `writeback` (LLM-derived).

**Import discipline:** stub modules defer heavy/optional imports (numpy, requests, hnswlib, sklearn,
indic-transliteration) — they use `if TYPE_CHECKING:` for type-only references — so every `src` module
imports with only the core deps present. Keep this property; it keeps tests fast and offline.

### Normalization design notes (Phase 1)

`normalize(text, cfg)` runs a fixed, deliberate rule order (see the module docstring): NFC → Devanagari
numerals → optional transliteration → lowercase → typed-placeholder replacement
(URL → EMAIL → AMOUNT → DATE → CARD/PHONE → MONTH → ID → NUM) → optional punctuation strip. Decisions a
future editor should know before changing it:
- **Card vs phone vs amount by length/pattern, not blunt stripping.** AMOUNT only matches
  currency-tagged numbers; a single regex finds digit *runs* and classifies by digit count
  (13–19 → `<CARD>`, 10–12 → `<PHONE>`); short/bare numbers fall through to `<NUM>`.
- **Allowlist semantics.** `meaningful_tokens_allowlist` (config) holds canonical numeric/word tokens
  (`0`, `0.00`, `twice`, …). AMOUNT/NUM compare the numeric core against it and, if matched, leave the
  token verbatim (the number IS the signal). Note `$0.00` keeps `0.00` but drops the `$` (currency
  symbols are stripped); add `$0.00` to the allowlist only if you also protect the symbol.
- **Two multilingual gotchas (already handled — don't regress):** (1) punctuation stripping keeps Unicode
  **Letters/Marks/Numbers** by category, because `\w` does *not* match Devanagari combining matras and a
  `\w`-based strip shreds Hindi; (2) lowercasing preserves `<TAG>` tokens, which keeps `normalize()`
  idempotent.
- **Conservative-by-design:** the month list excludes the ambiguous English "may"; `<ID>` requires a
  letter plus a ≥3-digit run (length ≥6) to avoid eating words like "covid19". Both are easy to widen.
- **Transliteration is off by default** and, when on, uses an imperfect `indic-transliteration` (ITRANS)
  fallback; production Hinglish→Devanagari needs a model (AI4Bharat IndicXlit). Tests for it
  `importorskip`.

### Jina client & stores design notes (Phase 2)

- **Network seam for offline tests.** `JinaClient(cfg, session=...)` accepts an injected `requests`
  session, so all tests fake the HTTP layer — no network, no key. Production builds a real
  `requests.Session`. The API key is read from `JINA_API_KEY` at construction; calls fail with
  `JinaAPIError` if it's missing.
- **Retry strategy (deviation from the plan, noted intentionally).** Uses a **manual exponential-backoff
  loop** rather than a urllib3 Retry adapter, because it makes transient-vs-fatal explicit (retry on
  connection errors + HTTP 429/5xx; other 4xx are fatal) and makes "raise `JinaAPIError` → pipeline skips
  write-back" clean to test. Same guarantees the plan asked for (batch, timeout, retry+backoff, graceful
  failure).
- **Embed cache key** = `model|task|dim|normalize|text`, so changing any of those does not return stale
  vectors. Vectors are L2-normalized **before** caching; the cache stores final vectors as float32 BLOBs.
- **Vector store scoring.** hnswlib returns `distance = 1 - similarity` for both `cosine` and `ip`
  spaces, so `Neighbor.score = 1 - distance` is the cosine similarity. `k` is clamped to the index size;
  the index auto-doubles `max_elements` when full. Metadata (id→intent/source/timestamp) lives in a
  SQLite sidecar.
- **Write-back safety lives in `ExactStore.put`** (see its docstring): seed is never overwritten; a
  writeback never flip-flops an existing intent; only a `seed` put overrides a prior `writeback`; every
  conflict is recorded in a `conflicts` table. The pipeline (P3) relies on this, plus the `T_write` gate
  and the audit log, for the §6 guarantees.

### Pipeline design notes (Phase 3)

`Pipeline.classify(query, history)` runs the query-only flow (`Pipeline(cfg, jina=, exact=, vector=,
llm=)` injects deps so tests run offline):
- **Embed once, reuse.** The semantic step embeds the normalized query a single time and reuses that
  vector for write-back — a full miss costs exactly one Jina round-trip, not two.
- **Graceful failure.** If `embed_one` raises `JinaAPIError`, the semantic layer is skipped, the LLM is
  called, and **write-back is skipped** (we have no vector and the API is down, §2).
- **OOS gate then write-back gate.** If LLM confidence `< oos_confidence_floor` → return `fallback`
  (`is_oos=True`), never cached. Else if confidence `≥ T_write` and the intent isn't `fallback` → write
  back. A vector is added only when `ExactStore.put` reports `written` (so conflicts/dupes don't pollute
  the index). Every write-back attempt is appended to the JSONL audit log.
- **Query-only namespace for now.** Phase 3 keys the exact store with `prev_intent=None`; the
  `(query, previous_intent)` escalation path + adaptive `T_low` arrives in Phase 4. `Classification.source`
  ∈ {exact_cache, semantic_cache, llm_fallback} (head added in P5); `is_oos = (intent == fallback)`.
- **The LLM stub** (`StubLLMClassifier`) resolves via explicit `mapping` (raw query → intent/confidence,
  for deterministic tests) → keyword rules → low-confidence `fallback`. The real classifier just
  implements `LLMClassifier.classify` and is selected by `config.llm_fallback.impl`.

### History escalation design notes (Phase 4)

`classify(query, history, previous_intent)` tries the broad query-only path first, then escalates:
- **Namespacing prevents cross-contamination.** Every vector carries a `prev_intent` namespace
  (`""` = query-only). `VectorStore.query(prev_intent=X)` matches *only* that namespace, so the
  query-only lookup can never match a context-conditioned write-back (e.g. "yes do it" stored after
  card_lost), and vice versa. Exact-store namespacing was already in place via the composite key.
- **Escalation order:** query-only exact → query-only semantic (`T_high`) → exact `(query, prev_intent)`
  → semantic namespaced by `prev_intent` at the **adaptive** `T_low`. Escalation only runs when
  `history.mode != off` and a `previous_intent` is supplied.
- **Adaptive `T_low`** (`_effective_t_low(frac)`): `t_low` when only the previous intent is used; it
  slides toward `t_low_floor` as more history is folded in (`frac` = turns used / `window_turns`).
- **Optional window fusion** (`history.mode == prev_intent_window`): the last `window_turns` turns are
  normalized, embedded separately, and late-fused with the query vector (`fusion_weight`); this also
  raises `frac`, lowering the threshold. Window-embed failure degrades gracefully to query-only fusion.
- **Write-back namespacing.** Referential/short queries (≤ `history.referential_max_tokens` tokens) are
  written back under `previous_intent`; longer self-contained queries are written query-only — so
  "yes do it" doesn't pollute the global cache but a novel balance phrasing stays broadly reusable.

### Head + combiner design notes (Phase 5)

- **Where the head runs.** As pipeline step 4 — *after* the cache layers (exact + semantic NN, query-only
  and escalation) miss their thresholds, and *before* the LLM. It is the learned generalization layer.
- **Head** (`head.py`): logistic regression (default) or nearest-centroid over the cached embeddings.
  `predict_proba = softmax(logits / T)`; for logreg `logits = log(clf.predict_proba)`, for centroids
  `logits = cosine sims`. Temperature `T` (config: `combiner.calibration`) is fit on a held-out set by
  NLL minimization so the head's confidence is comparable to the thresholds; `T = 1` = uncalibrated.
- **Combiner** (`combiner.py`) returns a `Decision` or `None` (None ⇒ defer to the LLM): `nn_only`
  disables the head; `head_only` accepts the head's argmax if `≥ combiner.head_threshold`; `nn_then_head`
  additionally reinforces confidence to `max(head_prob, nn_score)` when a sub-threshold NN neighbor
  *agrees* with the head — the cascade. The `fallback`/OOS outcome still comes from the pipeline's LLM +
  OOS gate, not the combiner.
- **Adding an intent** = refit the head on cached embeddings (seconds) or add a centroid — never retrain a
  base model (§13). The head is optional: if no `combiner.head_path` artifact exists and none is injected,
  the pipeline behaves exactly as Phase 4.

### Calibration / safety / eval design notes (Phase 6)

- **Scripts are split into a testable core + a thin `main()`.** The core functions take injected deps
  (embedder, stores, pipeline) so the whole suite runs offline with `MockEmbedder`; `main()` wires the
  real `JinaClient`. `MockEmbedder` has no semantics, so **calibration/eval numbers are only meaningful
  with `JINA_API_KEY` set** on the work laptop — the mock still exercises exact-cache growth.
- **Threshold calibration** (`calibrate_thresholds.py`): records nearest-neighbor similarity + correctness
  on a held-out split, then `sweep_thresholds` → hit-rate / false-hit curves and `recommend(curve,
  max_false_hit)` picks the lowest threshold within the false-hit budget (max coverage at acceptable
  risk). T_high uses query-only NN; T_low uses prev-intent-namespaced NN. **Copy the recommended values
  into `config.thresholds` before production.**
- **Error-amplification** (`simulate_growth.py`): a `NoisyOracle` (wrong w.p. `noise`, less confident when
  wrong) drives write-back; sweeping `T_write` shows the tradeoff — lower gate ⇒ higher hit-rate but
  higher cache-false-hit. This is how `T_write` is chosen (e.g. the offline run picks ~0.90: high accuracy,
  zero cached errors). Run it to set `T_write` for your data/oracle.
- **Audit/safety** (`audit_cache.py` + `ExactStore.conflicts/purge_writebacks`): inspect per-intent/source
  counts and recorded conflicts; purge write-back entries (seed is never purged) to roll back bad growth.
- **Evaluation** (`evaluate.py`): accuracy, macro-F1, hit-rate + exact/semantic/head split, false-hit rate,
  per-language (en/hi/hinglish) accuracy, OOS precision/recall, and p50/p95 latency **per source** — so
  the honest latency story (exact = sub-ms, semantic = one API round-trip) is measured, not assumed.
- **Serving** (`serve.py`): `POST /classify` over `Pipeline`; fastapi is optional and imported lazily.

---

## 4. Config reference (`config/config.yaml`)

All tunables live here; **no magic numbers in code**. Sections:

- **`jina`** — `api_url`, `model` (`jina-embeddings-v3`), `task` (`classification`), `dimensions` (256,
  must equal `vector_store.dim`), `normalize` (true), `timeout_seconds`, `max_retries`,
  `backoff_base_seconds`, `batch_size`, `embed_cache_path`.
- **`thresholds`** — `t_high`, `t_low`, `t_low_floor`, `t_write`. **Placeholders** until calibrated.
- **`normalization`** — `lowercase`, `strip_punctuation`, `devanagari_numerals`,
  `transliterate_romanized` (off; adds latency), `placeholders.{amount,card,phone,date,month,num,email,
  url,id}`, `meaningful_tokens_allowlist`.
- **`history`** — `mode` (`off|prev_intent|prev_intent_window`), `window_turns`.
- **`combiner`** — `mode` (`nn_only|head_only|nn_then_head`), `head_type` (`logreg|nearest_centroid`),
  `calibration` (`none|temperature`), `head_path`.
- **`vector_store`** — `backend` (`hnswlib|faiss`), `space` (`cosine`), `dim`, `max_elements`,
  `ef_construction`, `ef_search`, `M`, `index_path`, `meta_path`.
- **`exact_store`** — `backend` (`sqlite|json`), `path`.
- **`eviction`** — `ttl_days`, `max_entries`, `protect_seed`. **`audit`** — `log_path`,
  `record_conflicts`.
- **`llm_fallback`** — `impl` (`stub|…`), `stub_default_confidence`, `oos_confidence_floor`.
- **`oos`** — `fallback_intent` (`fallback`). **`paths`** — `intents`, `seed_csv`, `data_raw`,
  `models_dir`. **`seed`** — 42.

Invariants enforced by `tests/test_config.py`: `jina.dimensions == vector_store.dim`; `jina.task ==
classification`; thresholds in (0,1) with `t_low_floor ≤ t_low`; `seed == 42`; `oos.fallback_intent`
exists in the taxonomy; no `api_key` in config.

---

## 5. Data contract (drop your real data in on the work laptop)

Training/eval run on the user's machine. This repo ships a **synthetic** placeholder; replace it with
real data without code changes.

**`data/intents.yaml`** — the taxonomy. Shape:
```yaml
intents:
  <intent_name>:
    description: "<one line>"
```
Includes deliberately **confusable clusters** (cards / transactions-disputes / transfers) and an
explicit **`fallback`** (OOS) intent. To add an intent, add a key here (then seed it — see *How to
extend*).

**`data/seed/seed.csv`** — seed examples embedded into the cache. Columns:
```
query,intent,lang,history,prev_intent
```
- `query` — raw utterance (quote it; may contain commas; UTF-8 for Devanagari).
- `intent` — must exist in `intents.yaml`.
- `lang` — `en | hi | hinglish` (used for per-language metrics).
- `history` — usually empty; for referential rows, prior turns joined with ` [SEP] `.
- `prev_intent` — usually empty (query-only namespace); set it for **referential** rows so the seed is
  stored under that context (e.g. "yes do it" with `prev_intent=card_lost_or_stolen`). This is what makes
  the escalation path resolve the same surface query differently by context.
The synthetic file includes EN/HI/Hinglish rows, confusable pairs, multi-entity normalization cases
(card vs phone vs amount, `$0.00`, "twice"), and namespaced referential rows.

**To use real data on the work laptop:**
1. `pip install -r requirements.txt`; `export JINA_API_KEY=…`.
2. Put your labeled CSV in `data/raw/` (git-ignored) and point `paths.seed_csv` at it (same schema), or
   replace `data/seed/seed.csv`. Update `data/intents.yaml` to your taxonomy.
3. `python scripts/seed_cache.py` → seeds exact + vector stores (source=seed) and trains/saves the head.
4. `python scripts/calibrate_thresholds.py` → **derive** T_high/T_low from the sweep; copy the recommended
   values into `config.thresholds`.
5. `python scripts/simulate_growth.py` → pick `T_write` from the error-amplification tradeoff; copy it in.
6. `python scripts/evaluate.py` → metrics (accuracy, macro-F1, hit-rate, false-hit, per-language, OOS,
   latency-per-path). `python scripts/audit_cache.py [--purge]` → review/roll back cache growth.
7. (optional) `uvicorn src.serve:app` after `app = create_app()` for the `POST /classify` endpoint.

Built artifacts (stores, index, embed cache, head, audit log) land in `models/` (git-ignored).

---

## 6. How to extend

- **Add an intent (no base-model training):** add it to `data/intents.yaml`; add seed rows; then either
  refit the head (`train_head`, seconds) / add a nearest-centroid, or just add seed cache entries via
  `seed_cache.py`. The encoder (Jina) is fixed — adapters/keys are remote and untouched.
- **Plug in the real LLM:** implement `LLMClassifier` (subclass in `src/llm/`), return
  `LLMResult(intent, confidence)`, and select it via `config.llm_fallback.impl`. Nothing else changes.
- **Swap the vector store:** `pip install faiss-cpu`, set `vector_store.backend: faiss`, implement the
  faiss path behind the existing `VectorStore` interface. Same for `exact_store.backend: json`.
- **Tune behaviour:** everything is in `config.yaml` — history mode, combiner mode, normalization
  toggles, eviction caps, thresholds.

---

## 7. Phase status, limitations, open questions

**Done — Phase 0:** structure, pinned `requirements.txt`, full `config.yaml`, `intents.yaml`, synthetic
multilingual `seed.csv`, functional `config.py`, stub modules+scripts with documented contracts, smoke
+ config-sentinel tests, README/PROJECT_CONTEXT/CLAUDE.

**Done — Phase 1:** `preprocess/normalize.py` — typed-placeholder multilingual normalization (EN/HI/
Hinglish) with the meaningful-token allowlist; 30+ unit tests in `tests/test_normalize.py`.

**Done — Phase 2:** `embeddings/jina_client.py` (batched, cached, retrying, offline-mockable),
`store/exact_store.py` (SQLite + write-back safety), `store/vector_store.py` (hnswlib + SQLite meta);
`tests/test_jina_client.py`, `tests/test_exact_store.py`, `tests/test_vector_store.py`.

**Done — Phase 3:** `llm/fallback.py` (`LLMClassifier` interface + `StubLLMClassifier`) and `pipeline.py`
(query-only flow: exact → semantic(T_high) → LLM miss → OOS gate → confidence-gated, audited write-back,
with graceful Jina-failure handling); `tests/test_pipeline.py`.

**Done — Phase 4:** previous-intent escalation in `pipeline.py` + `prev_intent` namespacing in
`vector_store.py` (adaptive `T_low`, optional window late-fusion, namespaced write-back for referential
queries); `tests/test_history.py`.

**Done — Phase 5:** `classifier/head.py` (logreg / nearest-centroid + temperature calibration) and
`classifier/combiner.py` (nn_only / head_only / nn_then_head cascade), wired as the head stage in
`pipeline.py`; `tests/test_head.py`, `tests/test_combiner.py`, + pipeline head-stage tests.

**Done — Phase 6:** `embeddings/mock.py`; the scripts `seed_cache.py`, `calibrate_thresholds.py`,
`simulate_growth.py`, `audit_cache.py`, `evaluate.py` (testable core + thin `main()`); `src/serve.py`
(optional FastAPI); `ExactStore.conflicts/purge_writebacks`; `tests/test_scripts.py`,
`tests/test_serve.py` (skips without fastapi). **Full suite: 128 tests (127 passing, 1 skipped), ~3s.**

**Remaining = operational, not code** (run on the work laptop with `JINA_API_KEY`): point config at real
data, `seed_cache.py`, `calibrate_thresholds.py` → copy derived T_high/T_low into config,
`simulate_growth.py` → set T_write, `evaluate.py` → metrics. Possible future work: hnswlib native filtered
search for very large caches; a real `LLMClassifier`; IndicXlit for high-quality Hinglish transliteration.

**Infrastructure defaults chosen (swappable):** vector store = hnswlib; exact/embed/audit store =
sqlite (stdlib); HTTP = requests with a **manual exponential-backoff loop** (chosen over a urllib3 Retry
adapter for explicit transient/fatal handling + testable graceful failure — see Phase 2 design notes);
Matryoshka dim = 256.

**Known limitations / watch-list:**
- Thresholds in config are **placeholders** — do not ship them; derive in P6.
- VNNI/hardware latency caveats from v1 do **not** apply here (embedding is remote), but the **network
  round-trip dominates** semantic-path latency — measure it for real in P6.
- No large public Romanized-Hinglish banking set exists; the seed file is synthetic. Real Hinglish
  coverage depends on the user's data + optional transliteration.

**Open questions for the user (revisit as phases land):** real LLM interface details (sync/async, batch,
exact confidence semantics); whether to enable `transliterate_romanized` by default (latency vs
cross-script hit-rate); expected cache size / eviction policy at production scale.

---

## 8. Conventions (mirrored from ../v1)

`from __future__ import annotations` everywhere; module docstrings with context; full type hints
(`A | B`); `@dataclass` for domain objects; config as a plain dict (no Pydantic); exact-pinned
`requirements.txt` with optional packages marked; `pytest` with config-sentinel + import smoke tests;
seeds fixed at 42. Keep modules small and single-purpose.
