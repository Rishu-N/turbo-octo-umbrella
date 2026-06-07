# CLAUDE.md

Persistent context for Claude Code. Read this fully before writing any code. These decisions are the result of prior research and are intentional — do not silently deviate. If you believe a decision is wrong, say so explicitly and wait for confirmation before changing it.

---

## 1. What we are building

A **fine-grained intent classifier for a banking chatbot**. Given the **conversation history + the current user message**, classify the user's intent into one of a fixed (but growing) set of intents, or into a **fallback / out-of-scope (OOS)** bucket.

- Starts at ~20 intents, **will scale up**; **new intents are added frequently (roughly weekly)**.
- Intents are **semantically close / confusable** (e.g. "card not working" vs "card payment declined"). Separating look-alikes is the core difficulty.
- **Class imbalance**: some intents have several thousand examples, some ~1,000.

## 2. HARD CONSTRAINTS (never violate)

These are non-negotiable. Every design and code decision must respect them.

1. **Latency: 100 ms END-TO-END on CPU**, single request (batch size 1). End-to-end = preprocessing + tokenization + embedding + classification + OOS gate. Not just model inference. Leave headroom (target ≤ ~60 ms compute so network/preprocess fit under 100 ms).
2. **CPU-only inference, ON-PREM.** No GPU in production. **No external/3rd-party API calls at inference time.** All models and data are downloaded at build time and cached locally.
3. **Encoder + lightweight classifier ONLY on the hot path.** **Do NOT use a decoder / generative LLM that autoregressively generates the intent name** — it will miss the latency budget. (A small decoder may later be added *off* the hot path as an async fallback, but not now and not inline.)
4. **Adding a new intent must NOT require retraining the encoder.** It should be a cheap head refit (seconds–minutes) and/or a prototype/centroid update. The encoder is treated as frozen at the architectural level.
5. **Languages: English-dominant, but Hindi (Devanagari) AND Hinglish (Romanized / code-mixed) must work and must not be dropped.** Romanized/code-mixed is the hardest case.
6. **Commercial licensing.** This is a banking product. Only use models/datasets that are commercially usable on-prem. See §9.

## 3. Architecture (the system to build)

Two interchangeable approaches sharing ~90% of the code and an identical inference path:

- **Approach A — Frozen embeddings + Logistic Regression head.** Pre-trained sentence encoder used as-is (never fine-tuned) → fixed vector → logistic-regression head → intent probabilities. Fastest to ship; new intents = refit head.
- **Approach B — SetFit + Logistic Regression head.** Same encoder, first **contrastively fine-tuned** (SetFit) so confusable intents are pulled apart in embedding space, then a logistic-regression head. Same runtime cost as A (only offline training differs). Expected to win on confusable banking pairs.

Build A first as the baseline/floor; add B as the upgrade and compare. **The confusion matrix is the key artifact** for deciding A vs B and for finding colliding/duplicate intents.

### Inference pipeline (hot path)
1. **Preprocess**: optional language-ID; optional Romanized-Hindi → Devanagari transliteration (config flag); build input = last **1–2** history turns + current message, joined with ` [SEP] `, truncated to **64–96 tokens**. For e5 models, prefix with `query: `.
2. **Encode**: one forward pass through the (ONNX, int8-quantized) frozen encoder → vector. Cache prior-turn embeddings where the design allows.
3. **Classify**: logistic-regression head → calibrated probabilities over intents.
4. **OOS gate**: if confidence/distance fails the calibrated threshold → return `fallback` (none/human-handoff). See §6.
5. **Return**: `{intent, confidence, is_oos}`.

### Training (offline)
- **A**: encode all labeled examples once (cache embeddings) → fit `LogisticRegression(class_weight="balanced")`.
- **B**: SetFit contrastive fine-tune on labeled pairs → fit LR head → **export the fine-tuned encoder to ONNX + int8**.
- Calibrate the OOS threshold on the dev set (jointly maximize in-scope macro-F1 and OOS recall).
- Optionally compute per-class **prototypes (centroids)** for distance-based OOS and instant new-intent addition.

## 4. Recommended models (start here)

| Role | HF repo ID | Params | Dim | Max tok | License | Notes |
|---|---|---|---|---|---|---|
| **PRIMARY** (default A encoder & B backbone) | `intfloat/multilingual-e5-small` | ~118M | 384 | 512 | MIT | Best speed/quality for CPU; quantizes to ~113 MB int8. **Requires `query:` / `passage:` prefixes.** |
| **QUALITY LEVER** (best SetFit backbone) | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | ~278M | 768 | 128 | Apache-2.0 | Use if accuracy gain beats latency cost. |
| **HINGLISH TAIL** (routed) | `l3cube-pune/indic-sentence-bert-nli` | ~236M (inferred) | 768 | (MuRIL=512) | CC-BY-4.0 | MuRIL-based; handles Devanagari + transliterated text. Deploy for the Hinglish slice if that tail underperforms. |

Default to the PRIMARY model. Make the model name a **config value** so it can be swapped without code changes.

## 5. Data

**Bring-your-own-data is primary**: the production system trains on *our own* labeled banking utterances and *our own* intent taxonomy. Public datasets below are for **bootstrapping, benchmarking, OOS calibration, and the Hindi/Hinglish tail** — not the final label space.

| Dataset | HF repo ID | Size | Intents | Lang | License | Use |
|---|---|---|---|---|---|---|
| BANKING77 | `PolyAI/banking77` | 13,083 | 77 (fine-grained banking) | EN | CC-BY-4.0 | Benchmark the approach; sanity-check confusable-intent separation. |
| CLINC150 | `clinc_oos` (config `plus`) | 23,700 + OOS | 150 + OOS | EN | CC-BY-SA 3.0 | **OOS calibration** (has out-of-scope examples). |
| MASSIVE | `AmazonScience/massive` (`hi-IN`) | ~11.5k/locale | 60 | HI + 50 more | CC-BY-4.0 | Hindi labeled data for the tail. |
| HWU64 | (HF mirrors) | ~25,716 | 64 | EN | CC-BY-SA 3.0 | Extra multi-domain eval. |

> **Licensing note:** CLINC150 and HWU64 are **CC-BY-SA (share-alike)**. Keep them as **evaluation/calibration data only**; do **not** bake them into a redistributed model artifact, to avoid copyleft questions.

**Hinglish data:** no large public Romanized-Hinglish banking intent set exists. Plan to (a) machine-translate + transliterate BANKING77 / our own data to Hindi/Hinglish and human-verify, and (b) lean on SetFit's few-shot strength to bootstrap from a handful of real Hinglish examples per intent.

## 6. Out-of-scope / fallback (must implement)

- Always provide an explicit **`fallback` / none** class.
- Implement OOS as a **config-selectable** strategy:
  - `threshold`: max softmax probability below a calibrated cutoff → fallback (simple baseline; softmax is overconfident).
  - `prototype`: Mahalanobis/cosine distance to nearest class centroid exceeds threshold → fallback (fits the embedding design; supports instant new intents).
  - `energy`: energy score over logits (stronger than raw softmax).
- **Calibrate the threshold on the dev set.** Report OOS recall/precision separately from in-scope accuracy.

## 7. Conversation history handling

- Use only the **last 1–2 turns** + current message. Research shows more context often *hurts* and costs latency.
- Join with ` [SEP] `; optionally prepend the **previous predicted intent** as a cheap, informative feature.
- Cap total tokens (64–96). Cache prior-turn embeddings where the architecture allows to avoid recompute.

## 8. CPU latency optimization (required, in this order)

1. **Export encoder to ONNX Runtime** with full graph optimization (`ORT_ENABLE_ALL`), done offline.
2. **Dynamic int8 quantization** via `optimum[onnxruntime]` using the **`avx512_vnni`** config. ~4× smaller, ~2–4× faster **on AVX-512-VNNI CPUs**.
3. **Truncate sequence length** to 64–96 tokens (cheapest lever).
4. **Single-request tuning**: intra-op threads = physical cores; batch=1; avoid thread oversubscription.
5. Minimize tokenization cost; skip padding for single requests.

> **CRITICAL hardware caveat:** int8 speedups depend heavily on **AVX-512-VNNI**. On a CPU without VNNI, quantization may give only ~25% speedup vs ~250% with it. **Benchmark latency on production-equivalent x86 VNNI hardware — NOT on the Apple Silicon dev machine.** Apple Silicon is for development only; keep dev/prod parity by standardizing on the **ONNX Runtime CPU execution provider** as the source of truth for latency numbers.

## 9. Tech stack & conventions

- **Python 3.10+**. Pin versions in `requirements.txt`. Set random seeds everywhere (default seed `42`) for reproducibility.
- Core libs: `sentence-transformers`, `transformers`, `scikit-learn`, `setfit`, `optimum[onnxruntime]`, `onnxruntime`, `datasets`, `numpy`, `pandas`.
- Transliteration: `indic-transliteration` or AI4Bharat IndicXlit. Language-ID: fastText `lid.176` or `l3cube-pune/hing-bert-lid`. Keep these **optional/config-gated**.
- Optional serving: `fastapi` + `uvicorn`. Optional ANN for prototypes: `hnswlib` or `faiss-cpu`.
- **All tunables live in `config/config.yaml`** (model name, seq len, history turns, OOS method/threshold, paths, thread count). No magic numbers in code.
- Type hints + docstrings on public functions. Keep modules small and single-purpose (see §10).
- **No network calls at inference.** Download/caching happens in `scripts/download_data.py` and model-prep steps only.
- Tests with `pytest`, including a **latency regression test** that fails if p95 exceeds the budget on the CI/benchmark host.

## 10. Project structure (target)

```
banking-intent-classifier/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/{raw,processed,intents}/
├── src/
│   ├── preprocess.py      # language-id, transliteration, history+message input building
│   ├── encoder.py         # load encoder, ONNX export, int8 quantize, embed()
│   ├── classifier.py      # LR head: train / save / load / predict_proba
│   ├── setfit_train.py    # Approach B training
│   ├── oos.py             # threshold | prototype | energy
│   ├── pipeline.py        # end-to-end inference (the hot path)
│   ├── train.py           # entrypoint: --approach {frozen,setfit}
│   ├── evaluate.py        # macro-F1, per-intent F1, OOS recall/prec, confusion matrix
│   └── serve.py           # optional FastAPI app
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py    # stratified train/dev/test, confusable pairs well represented
│   ├── benchmark_latency.py  # p50/p95/p99 @ batch=1, end-to-end
│   └── add_intent.py      # add an intent WITHOUT retraining the encoder
├── models/                # onnx encoders, LR heads, prototypes, thresholds
└── tests/
```

## 11. Key commands (wire these up as you build)

```bash
pip install -r requirements.txt
python scripts/download_data.py            # fetch public datasets, cache locally
python scripts/prepare_data.py             # stratified splits
python src/train.py --approach frozen      # Approach A
python src/train.py --approach setfit      # Approach B
python src/evaluate.py --split test        # metrics + confusion matrix
python scripts/benchmark_latency.py        # p50/p95/p99 end-to-end on CPU
python scripts/add_intent.py --name balance_inquiry --examples-file data/intents/balance_inquiry.txt
uvicorn src.serve:app --host 0.0.0.0 --port 8080   # optional
pytest                                     # unit + latency regression tests
```

## 12. Evaluation requirements (don't skip)

- **Quality**: accuracy, **macro-F1** (headline metric, because of class imbalance), **per-intent F1**, and a **confusion matrix** (the most important diagnostic — surfaces colliding intents).
- **OOS**: recall and precision, reported separately.
- **Latency**: **p50/p95/p99 at batch=1 on the production-equivalent CPU**, measured **end-to-end** (incl. tokenization, transliteration, head). Warm up before measuring. A regression test must fail if p95 > budget.
- **New-intent eval**: hold out an intent, add it via `add_intent.py` (head refit / SetFit), then measure both new-intent F1 **and** retention of old-intent F1 (catastrophic forgetting check).

## 13. Things NOT to do

- Do **not** put a generative/decoder model on the hot path.
- Do **not** fine-tune or require retraining the encoder to add an intent.
- Do **not** use `localStorage`/`sessionStorage` or any browser storage (N/A here, but no hidden state — keep state explicit and in config/model files).
- Do **not** trust latency numbers from the Apple Silicon dev box for the production budget.
- Do **not** bake CC-BY-SA datasets (CLINC150, HWU64) into a redistributed artifact.
- Do **not** add network calls to the inference path.
- Do **not** drop Hindi/Hinglish handling to chase English numbers.
