# Banking Intent Classifier (CPU · On-Prem · Multilingual)

A fast, on-prem intent classifier for a banking chatbot. It reads the **conversation history + the current user message** and predicts the user's intent, or routes to a **fallback / out-of-scope** bucket. Built to run **under 100 ms end-to-end on CPU**, handle **English, Hindi, and Hinglish**, and let you **add new intents frequently without retraining the encoder**.

> **Design in one line:** a frozen (or contrastively fine-tuned) multilingual sentence encoder turns each message into a vector; a lightweight logistic-regression head turns that vector into an intent. All the heavy understanding is in the encoder; all the cheap, frequently-updated decision-making is in the head.

---

## 📍 HANDOFF — READ THIS FIRST (current state)

> This section exists so a fresh LLM/engineer can pick up the project cold. It reflects what is **actually built and verified right now**, not the original aspirational plan. Read [`CLAUDE.md`](CLAUDE.md) for the full design constraints — they are non-negotiable. The authoritative phase plan lives at `~/.claude/plans/initial-prompt-for-linked-meadow.md`.

### Where we are

The project is being built in phases (see [Roadmap](#roadmap)). **What works end-to-end today:**

- **Approach A** — frozen `multilingual-e5-small` encoder → Logistic-Regression head.
- **Approach B** — **SetFit** (contrastive fine-tune of the encoder) → LR head. *This is "fine-tune on my data."*
- Stratified data prep, training (both approaches), and evaluation (accuracy, macro-F1, per-intent F1, confusion matrix).

All of the above has been **smoke-tested on synthetic data** (see [Data](#data-current-state)). It has **not** yet been run on real banking data, and the latency/OOS/serving pieces are **not built yet** (see [What's NOT built](#whats-not-built-yet-deferred)).

### What runs today (verified commands)

```bash
# 0. one-time environment setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py            # caches BANKING77 / CLINC150 / MASSIVE hi-IN to data/raw/

# 1. (synthetic placeholder data already committed; regenerate with:)
python scripts/make_synthetic_csv.py        # writes data/raw/user_dataset.csv

# 2. stratified split  →  data/processed/{train,dev,test}.parquet
python scripts/prepare_data.py

# 3a. TRAIN Approach A (frozen + LR head)
python -m src.train --approach frozen

# 3b. TRAIN Approach B (SetFit fine-tune + LR head)   ← the fine-tune path
python -m src.train --approach setfit

# 3c. (optional) also train the cross-encoder RERANKER on top of either stage-1
python -m src.train --approach setfit --with-reranker

# 4. EVALUATE on the test split  →  prints metrics, writes models/reports/
python -m src.evaluate --split test                  # stage-1 only
python -m src.evaluate --split test --reranker on     # stage-1 vs reranked, side by side

# tests (config sanity + smoke)
pytest
```

> **Important:** modules are run as packages — `python -m src.train`, **not** `python src/train.py` (the latter breaks the relative imports). Set `TOKENIZERS_PARALLELISM=false` to silence a harmless HF warning during SetFit.

### Two environment gotchas already solved (don't re-break these)

1. **`huggingface_hub` is pinned to `0.23.5`.** `setfit==1.0.3` imports `DatasetFilter`, which was removed in `huggingface_hub>=0.24`. 0.23.5 is the last version with it and still satisfies `transformers`/`sentence-transformers`/`datasets`. If you bump `setfit`, you can likely unpin.
2. **`fasttext-wheel` instead of `fasttext`.** `fasttext==0.9.2` needs a C++/pybind11 source build that fails on Python 3.11+. `fasttext-wheel==0.9.2` is the prebuilt drop-in. (fastText is optional/config-gated anyway — only used if `preprocess.language_id` is turned on later.)

### The golden rule for this codebase

**Every tunable lives in [`config/config.yaml`](config/config.yaml). No magic numbers in code.** Modules read config; CLI flags only *override* config. Seeds default to 42. If you add a knob, add it to config first.

---

## Why this design

The hard problems for a banking bot are (1) **confusable intents** ("card not working" vs "card payment declined"), (2) a **growing intent set** with new intents added often, (3) a **strict CPU latency budget**, and (4) **multilingual traffic** that is mostly English but includes Hindi and code-mixed Hinglish.

A monolithic fine-tuned transformer struggles with (2) — every new intent means a full retrain. Putting a small classifier on top of a sentence encoder decouples the two: the encoder stays fixed, and adding an intent is a seconds-long head refit (or a single centroid update). That is the central reason for this architecture.

We support **two interchangeable approaches** that share the same inference path:

| | **Approach A — Frozen embeddings + LogReg** | **Approach B — SetFit + LogReg** |
|---|---|---|
| Encoder | Used as-is (never fine-tuned) | Contrastively fine-tuned on your intents, then frozen |
| Build effort | Lowest (ship first) | Moderate (the upgrade) |
| Confusable intents | Decent | **Stronger** — contrastive training pulls look-alikes apart |
| New intents | Refit head (instant) | Refit head; periodic SetFit refresh |
| Inference latency | Meets budget | **Identical to A** (only offline training differs) |
| Status | ✅ built | ✅ built |

Build A as the baseline, then add B and compare on the **confusion matrix**. That matrix is the most important artifact in the repo — it tells you whether to move from A to B, merge two intents that are really the same, or add hard-negative examples.

## Models

Defaults are chosen for an English-dominant CPU deployment that must still cover Hindi/Hinglish. The encoder is a **config value** (`encoder.model_name`) and can be swapped without code changes.

- **Primary (default):** [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) — ~118M params, 384-dim, MIT. Smallest/fastest viable multilingual encoder; quantizes to ~113 MB int8. Requires `query:` / `passage:` prefixes (handled in `preprocess.py`).
- **Quality lever:** [`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) — ~278M, 768-dim, Apache-2.0. The best-documented multilingual SetFit backbone; set it via `setfit.backbone` when the accuracy gain justifies the latency cost.
- **Hinglish tail (routed, not yet wired):** [`l3cube-pune/indic-sentence-bert-nli`](https://huggingface.co/l3cube-pune/indic-sentence-bert-nli) — MuRIL-based, 768-dim, CC-BY-4.0. For the Hinglish slice if that tail underperforms.

> Models are downloaded from Hugging Face the first time `Encoder` loads them, then cached locally (`~/.cache/huggingface`). This is the only model-download step; **the inference path makes no network calls.**

## Data

### The real target dataset

Production trains on **your own labeled banking utterances and your own taxonomy**. The expected shape (per the user who owns this project):

- **~1,500 examples, 17 intents**, highly imbalanced (**8 → 243 examples per intent**).
- **3 CSV columns** (names configurable in `config.data.columns`):
  - `current_user_query` — the message to classify
  - `conversation_history` — prior turns (may be empty)
  - `expected_intent` — the gold label

To use it: drop the CSV at `data/raw/user_dataset.csv` (or set `config.data.user_dataset_path`) and run `scripts/prepare_data.py`. The extreme imbalance is why `class_weight=balanced` and the tiny-class split safeguard (below) exist, and why **per-intent F1** matters more than accuracy here.

### Data (current state)

The repo currently ships a **synthetic placeholder** so the pipeline is runnable without real data:
`scripts/make_synthetic_csv.py` → `data/raw/user_dataset.csv` (6 toy intents, ~79 rows, deliberately imbalanced). **Replace this with real data before drawing any conclusions** — current metrics are meaningless toy numbers that only prove the wiring works.

### Public datasets (scaffolding — cached by `download_data.py`)

For benchmarking, OOS calibration, and the Hindi/Hinglish tail — **not** the final label space:

- `PolyAI/banking77` — 77 banking intents, EN, CC-BY-4.0. Benchmark + confusable-pair check. *(train=10003, test=3080)*
- `clinc_oos` (config `plus`) — 150 intents **+ out-of-scope examples**, EN, CC-BY-SA 3.0. For **OOS threshold calibration** (Phase 2). *(train=15250, val=3100, test=5500)*
- `AmazonScience/massive` (`hi-IN`) — Hindi utterances, CC-BY-4.0. Tail coverage. *(train=11514, val=2033, test=2974)*
- HWU64 — optional, `--include-hwu64`, CC-BY-SA 3.0.

> CLINC150 and HWU64 are **share-alike (CC-BY-SA)** — evaluation/calibration only; don't redistribute them inside a model artifact.

## How it works

```
history + current message
        │
        ▼
 preprocess.py  ── language-id (optional, stub) ── Romanized→Devanagari (optional, stub)
        │      build "query: {history} [SEP] {current message}", tokenizer truncates to max_seq_length
        ▼
  encoder.py (SentenceTransformer; frozen or SetFit-fine-tuned)  →  fixed 384-dim vector
        │
        ▼
 classifier.py  (LogisticRegression, class_weight=balanced)  →  intent probabilities
        │
        ▼
   OOS gate (NOT BUILT YET — Phase 2)  →  intent  or  fallback
```

- **Conversation history:** only the last 1–2 turns are used (`history.num_turns`); more context tends to hurt accuracy and costs latency.
- **Out-of-scope:** designed but not yet implemented. Will be `threshold | prototype | energy`, calibrated on a dev set.

## Project structure (actual)

`✅ = built & working · ⏳ = planned, not yet created`

```
RAS/
├── CLAUDE.md                  ✅ persistent design context + HARD CONSTRAINTS (read first)
├── README.md                  ✅ this file
├── requirements.txt           ✅ pinned deps (note the huggingface_hub & fasttext-wheel pins)
├── .gitignore                 ✅
├── config/
│   └── config.yaml            ✅ ALL tunables — the single source of truth
├── data/
│   ├── raw/                   ✅ user_dataset.csv (synthetic) + cached public datasets
│   ├── processed/             ✅ train/dev/test.parquet (from prepare_data.py)
│   └── intents/               ✅ (empty; for per-intent example files used by add_intent.py later)
├── src/
│   ├── __init__.py            ✅
│   ├── config.py              ✅ load_config(), repo_path() — config loader used everywhere
│   ├── data.py                ✅ LabelMap, load_user_dataset(), to_xy(), describe()
│   ├── preprocess.py          ✅ build_input() / build_inputs() — history+query, e5 prefix
│   ├── encoder.py             ✅ Encoder class, embed(); backend hook for ONNX (Phase 3)
│   ├── classifier.py          ✅ HeadBundle, train_head/predict_proba/save_head/load_head
│   ├── setfit_train.py        ✅ train_setfit() — Approach B contrastive fine-tune
│   ├── train.py               ✅ CLI: --approach {frozen,setfit}
│   ├── evaluate.py            ✅ CLI: --split; metrics + confusion matrix
│   ├── pipeline.py            ⏳ end-to-end hot path (preprocess→encode→classify→OOS)
│   ├── oos.py                 ⏳ threshold | prototype | energy
│   └── serve.py               ⏳ FastAPI POST /classify
├── scripts/
│   ├── download_data.py       ✅ fetch+cache public datasets
│   ├── make_synthetic_csv.py  ✅ generate the synthetic placeholder CSV
│   ├── prepare_data.py        ✅ stratified split w/ tiny-class safeguard
│   ├── benchmark_latency.py   ⏳ p50/p95/p99 @ batch=1 end-to-end
│   └── add_intent.py          ⏳ add an intent WITHOUT retraining the encoder
├── models/                    ✅ classifier.joblib, label_map.json, encoder_finetuned/, reports/
│   └── reports/               ✅ confusion_matrix_test.csv, metrics_test.json
└── tests/
    └── test_smoke.py          ✅ config-keys + import + latency-budget sentinel
```

### Module dependency graph

```
config.py ← everything (config loader)
data.py     ← train.py, evaluate.py, setfit_train.py, prepare_data.py   (LabelMap, CSV load)
preprocess.py ← prepare_data.py                                          (build input strings)
encoder.py  ← train.py, evaluate.py, setfit_train.py                     (embed)
classifier.py ← train.py, evaluate.py, setfit_train.py                   (LR head + (de)serialize)
setfit_train.py ← train.py                                              (Approach B)
```

### Data & artifact flow

```
user CSV (data/raw/user_dataset.csv)
   │  scripts/prepare_data.py   (load → build_inputs → stratified split)
   ▼
data/processed/{train,dev,test}.parquet   (cols: query, history, label, text)
   │  python -m src.train --approach {frozen|setfit}
   ▼
models/classifier.joblib        (LR head + LabelMap + meta)
models/label_map.json           (human-readable label↔id)
models/encoder_finetuned/       (SetFit only — the fine-tuned SentenceTransformer)
   │  python -m src.evaluate --split test
   ▼
models/reports/metrics_test.json + confusion_matrix_test.csv
```

The `meta` dict inside `classifier.joblib` records `approach` and `encoder_path`, so `evaluate.py` automatically loads the fine-tuned encoder for SetFit models and the base encoder for frozen ones.

## Dependencies

Python **3.10+** (developed/verified on **3.11.3**). Full pinned list in [`requirements.txt`](requirements.txt). Highlights:

| Package | Version | Role |
|---|---|---|
| `sentence-transformers` | 2.7.0 | encoder wrapper |
| `transformers` | 4.39.3 | backbone models |
| `setfit` | 1.0.3 | Approach B contrastive fine-tune |
| `huggingface_hub` | **0.23.5 (pinned)** | compat shim for setfit 1.0.3 — see Handoff gotchas |
| `scikit-learn` | 1.4.2 | LogisticRegression head |
| `datasets` | 2.18.0 | load public datasets |
| `numpy` / `pandas` | 1.26.4 / 2.2.1 | arrays / dataframes |
| `optimum[onnxruntime]` / `onnxruntime` | 1.19.0 / 1.17.3 | Phase 3 ONNX/int8 (installed, not used yet) |
| `pyyaml` | 6.0.1 | config |
| `fastapi` / `uvicorn` | 0.110.1 / 0.29.0 | Phase 6 serving (installed, not used yet) |
| `indic-transliteration` / `fasttext-wheel` / `hnswlib` | — | optional, config-gated; not used yet |
| `pytest` | 8.1.1 | tests |

## Configuration reference

[`config/config.yaml`](config/config.yaml) sections:

- `encoder` — `model_name`, `query_prefix` (e5 needs `"query: "`), `max_seq_length` (96), `backend` (`pytorch`; `onnx_*` is Phase 3).
- `history` — `num_turns` (2), `separator` (`" [SEP] "`).
- `approach` — `frozen | setfit` (default for tooling that doesn't take `--approach`).
- `setfit` — `backbone` (null → use `encoder.model_name`), `num_epochs`, `num_iterations`, `batch_size`, `body/head_learning_rate`. **CLI flags on `train.py` override these.**
- `data` — `user_dataset_path`, `columns` (the 3 CSV headers), `min_samples_per_split` (2 — tiny-class floor).
- `oos` — `method` (`threshold`), `threshold` (0.45). *Not wired yet.*
- `preprocess` — `language_id`, `transliterate_romanized` (both `false`; stubs).
- `training` — `class_weight` (`balanced`), `seed` (42), `test_size`/`dev_size` (0.15/0.15).
- `latency` — `budget_ms` (100). *Hard constraint; smoke test asserts it.*
- `paths` — every input/output location.
- `runtime` — ONNX thread settings for Phase 3.

## Adding a new intent (planned workflow — `add_intent.py` not built yet)

The architecture supports it; the script is Phase 5. The intended flow:
1. Drop example utterances into `data/intents/<intent_name>.txt`.
2. Run `add_intent.py` — refits the LR head (and/or adds a class prototype) in seconds; **the encoder is untouched**. SetFit refresh for very-low-data intents.
3. Re-run `evaluate.py` to confirm the new intent's F1 and that existing intents didn't regress.

## Latency (Phase 3 — not yet implemented)

The 100 ms budget is **end-to-end** (preprocess + tokenize + embed + classify + OOS), at **batch size 1**. Planned optimization stack: ONNX export with full graph optimization → dynamic int8 quantization (`avx512_vnni`) → sequence truncation (64–96 tokens) → single-request thread tuning.

> **Benchmark on production-equivalent x86 hardware with AVX-512-VNNI.** int8 speedups are large *with* VNNI and modest without it, and **Apple Silicon numbers won't reflect the production CPU** (this project is developed on a Mac). A latency regression test must fail the build if p95 exceeds the budget on the benchmark host. **Do not trust any latency number measured on the dev Mac.**

## Evaluation

`python -m src.evaluate --split {train,dev,test}` reports and persists:

- **Macro-F1** — headline metric (class imbalance makes plain accuracy misleading).
- **Per-intent precision/recall/F1** — critical for the 8-example tail.
- **Confusion matrix** → `models/reports/confusion_matrix_<split>.csv`, plus the top confused pairs printed to stdout.
- `models/reports/metrics_<split>.json` — machine-readable, for A-vs-B comparison.

OOS recall/precision and the p50/p95/p99 latency report are planned (Phases 2 & 3).

## Roadmap

Phased build with a checkpoint after each phase. **Plan file:** `~/.claude/plans/initial-prompt-for-linked-meadow.md`.

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding (tree, deps, config, `download_data.py`) | ✅ done |
| 1 | Approach A (frozen + LR) + eval harness | ✅ done |
| (B) | Approach B (SetFit fine-tune) — pulled forward at user request | ✅ done |
| 2 | OOS gate (`threshold`/`prototype`/`energy`) + CLINC150 calibration | ⏳ deferred |
| 3 | ONNX export + int8 + p50/p95/p99 latency + regression test | ⏳ deferred |
| 4 | A-vs-B comparison on real data (confusion-matrix diff on confusable pairs) | ⏳ pending real data |
| 5 | `add_intent.py` (+ retention check) + Hindi/Hinglish transliteration/LID | ⏳ deferred |
| 6 | FastAPI serving + final polish | ⏳ deferred |

### What's NOT built yet (deferred)

- **OOS / fallback gate** (`src/oos.py`) — no out-of-scope handling on predictions yet.
- **End-to-end pipeline object** (`src/pipeline.py`) — there's no single `predict(history, message)` hot-path object yet; training/eval call the modules directly.
- **ONNX / int8 / latency** — `encoder.py` only has the PyTorch backend; `benchmark_latency.py` doesn't exist. **No latency has been validated.**
- **Serving** (`src/serve.py`) — no API.
- **`add_intent.py`** — the no-retrain intent-add workflow.
- **Hindi/Hinglish** — `preprocess.py` has transliteration/LID **stubs only** (no-ops); not evaluated on MASSIVE/Hinglish yet.

## Decisions & deviations log (for the next maintainer)

- **Approach B was pulled forward** ahead of the original phase order because the user explicitly asked to "fine-tune on my data." A and B are both done; Phases 2/3/5/6 are not.
- **`huggingface_hub` pinned to 0.23.5** — see Handoff gotchas. Revisit if you upgrade `setfit`.
- **`fasttext` → `fasttext-wheel`** — prebuilt wheel; avoids a source build that fails on Py 3.11+.
- **`src/config.py` added** (not in the original CLAUDE.md §10 tree) as the shared config loader, so modules don't each re-parse YAML. Harmless, keeps things DRY.
- **`scripts/make_synthetic_csv.py` added** to keep the repo runnable before real data lands. Delete once real data is in use.
- **`min_samples_per_split: 2`** chosen over k-fold for the tiny-class safeguard (simpler; an 8-example class yields ~4 train / 2 dev / 2 test). Revisit if headline numbers on the smallest classes look too noisy.
- **Not a git repo yet** — no commits have been made. `git init` + first commit is up to the user.

## Setup (full)

```bash
git clone <your-repo-url> && cd RAS      # (not yet a git repo — see decisions log)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py          # downloads + caches public datasets locally
```

Development happens on Apple Silicon; **production is CPU (x86, on-prem)**. To keep dev/prod parity, ONNX Runtime's CPU execution provider will be the source of truth for latency once Phase 3 lands — see [Latency](#latency-phase-3--not-yet-implemented).

## Licensing

All default models and datasets are commercially usable on-prem. Models: e5-small (MIT), paraphrase-mpnet (Apache-2.0), IndicSBERT (CC-BY-4.0). Datasets: BANKING77 and MASSIVE (CC-BY-4.0); CLINC150 and HWU64 (CC-BY-SA 3.0 — **evaluation/calibration only**, do not redistribute inside an artifact).
