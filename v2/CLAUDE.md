# CLAUDE.md — v2 (semantic-cache intent classifier)

**Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) fully before writing any code.** It is the complete
design, module map, config reference, and data contract. This file is just the non-negotiable
guardrails so they are never lost.

This is a **different project from `../v1`** (which is a frozen-encoder + LR + SetFit + ONNX
classifier). v2 shares v1's *conventions* but no runtime code.

## Hard constraints (never violate)

1. **Jina v3 is API-only.** Every `embed()` is a remote call: batch, timeout, retry w/ backoff, and on
   failure fall through to the LLM and **skip write-back**. Only the exact-match layer is sub-ms — never
   claim sub-10 ms for the embedding path.
2. **Jina `classification` task adapter, Matryoshka dim 256, L2-normalize.** All config values.
3. **`JINA_API_KEY` from env only.** Never hardcode; never in `config.yaml`.
4. **Do not concatenate full history into the embedded text.** Key on the normalized query alone;
   escalate to *previous intent* only when T_high fails, and keep previous-intent in the key.
5. **Typed-placeholder normalization, not blanket digit-stripping.** Card vs phone vs amount by
   length/pattern; preserve the configurable meaningful-token allowlist ("0", "$0.00", "twice", …).
6. **Write-back safety:** gate on T_write, de-dup conflicts (don't flip-flop), TTL + size caps, seed
   entries never evicted, audit every write-back.
7. **Thresholds are derived, not guessed** (`scripts/calibrate_thresholds.py`, Phase 6). Values in
   `config.yaml` are placeholders.
8. **Adding an intent never retrains a base model** — refit the head / add a centroid / add seed
   entries. **No network calls except the isolated Jina client** (mocked in tests).

## Workflow

- All tunables in `config/config.yaml` (no magic numbers in code). Python 3.10+, seeds = 42, type
  hints + docstrings, `pytest`.
- **Build phase by phase; stop at each checkpoint for review.** Current state: **all phases (0–6)
  complete** — see the *Phase status* section of `PROJECT_CONTEXT.md`. Remaining work is operational
  (seed real data, derive thresholds, evaluate on the work laptop with `JINA_API_KEY`).
- **Keep `PROJECT_CONTEXT.md` updated at the end of every phase** so the handoff doc always matches the
  code — it is a first-class deliverable.
