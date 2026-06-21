# Classifying Banking Intents in 100 Milliseconds, on a CPU, in Three Languages

*The engineering story behind a deceptively hard machine-learning problem — and why the answer was to make the model do **less**, not more.*

---

## The question that isn't simple

A customer types into a banking chatbot:

> *"my card isn't working"*

What do they want? Pause on that. It could be:

- a **declined payment** ("my card was rejected at checkout")
- a **blocked card** ("you froze my card after that fraud alert")
- a **broken chip** ("the terminal can't read it")
- a **forgotten PIN** ("it won't let me in at the ATM")
- a **lost card** ("…because I can't find it")

Five different intents. Five different downstream actions — issue a replacement, lift a block, verify a transaction, reset a PIN, escalate to a human. One nearly identical sentence. And the customer is annoyed *right now*, so you have a few dozen milliseconds to get it right before the bot feels broken.

This is the whole problem in miniature. Intent classification sounds like a solved, freshman-level NLP task — until you put it inside a real bank, on real infrastructure, under real latency, in the languages people actually type in. Then every "obvious" solution quietly falls apart.

This is the story of the design choices that *didn't* fall apart.

---

## Four constraints that change everything

Before a single line of model code, four hard constraints defined the entire shape of the solution. They're worth stating plainly, because each one kills a popular approach.

**1. 100 milliseconds, end-to-end, on a CPU.**
Not GPU. Not "p50 of the model forward pass." The whole thing — text cleanup, tokenization, embedding, classification, the out-of-scope check — under 100 ms at batch size one, on a commodity server CPU. This is a chatbot; humans feel latency above ~100 ms as lag.

**2. On-prem. No external API calls at inference time.**
It's a bank. Customer messages cannot leave the building to hit some hosted model's API. Everything — every weight, every tokenizer — is downloaded at build time and runs locally, forever after, with the network cable metaphorically unplugged.

**3. The intent set grows every week.**
Product teams add new intents constantly — a new card product, a new dispute flow, a seasonal campaign. If adding one intent means *retraining the whole model overnight*, the system can't keep up with the business.

**4. English, Hindi, and Hinglish — all of them.**
Traffic is English-dominant but meaningfully Hindi (in Devanagari script) *and* Hinglish — romanized, code-mixed text like *"mera card block ho gaya hai"*. The last one is the cruel case: it's neither clean English nor clean Hindi, and there's almost no public training data for it.

Hold these four in your head. Now watch the two most popular 2020s answers die.

---

## Why the two "obvious" answers fail

**Obvious answer #1: Fine-tune a big transformer classifier.**
Take BERT-large or similar, bolt a classification layer on top, fine-tune end-to-end on your intents. This works great on benchmarks. It fails constraint #3 catastrophically: every new intent reshapes the output layer and means a full, expensive retrain. It also strains #1 — large transformers are not naturally 100 ms on CPU. You'd be retraining a slow model every Tuesday.

**Obvious answer #2: Just ask an LLM.**
"Here are the 17 intents, here's the message, which one is it?" A generative model autoregressively spells out the intent name. Beautiful in a demo. It detonates on constraint #1 (a decoder generating tokens, on CPU, in under 100 ms? no) and constraint #2 (the good ones are hosted APIs you can't call). Generation is the wrong tool: you don't need the model to *write*, you need it to *choose*.

Both failures point at the same insight. The expensive part — *understanding the sentence* — and the cheap, fast-changing part — *deciding which bucket it falls in* — are glued together when they shouldn't be.

So we unglue them.

---

## The core idea: a frozen brain and a cheap, swappable mouth

Here's the architecture in one breath:

> A **frozen multilingual sentence encoder** turns each message into a single fixed-length vector — a point in a ~384-dimensional "meaning space." Then a **tiny logistic-regression head** draws boundaries in that space to pick the intent.

```
"my card isn't working"
        │
        ▼
  [ sentence encoder ]   ← heavy, frozen, never changes
        │
        ▼
   [0.12, -0.83, ... ]   ← a 384-dim vector. The "meaning."
        │
        ▼
 [ logistic regression ] ← featherweight, retrained in seconds
        │
        ▼
   intent: card_declined  (confidence 0.71)
```

The mental model that makes this click: **the encoder is a multilingual semantic map.** Sentences that mean similar things land near each other — and crucially, this is true *across languages*. "I lost my card," "मेरा कार्ड खो गया," and "mera card kho gaya" should all cluster in the same neighborhood. A good multilingual encoder gives you that for free.

Once your sentences are points on a map, classification is just *"which region is this point in?"* — and logistic regression is a perfectly good, blazing-fast way to carve a map into regions.

This split is the whole game, because of what it buys you against the four constraints:

- **Constraint #3 (growing intents):** Adding an intent doesn't touch the encoder at all. You just re-draw the regions — refit the logistic head in *seconds*, or drop in one new "this is what the new intent looks like" centroid. The expensive brain is frozen; only the cheap mouth changes.
- **Constraint #1 (latency):** At inference you do exactly one encoder forward pass and one matrix multiply. No generation loop. That's the cheapest possible path to an answer.
- **Constraint #4 (multilingual):** Pick an encoder that was *pretrained* multilingual, and Hindi/Hinglish coverage comes from the frozen brain, not from labels you don't have.
- **Constraint #2 (on-prem):** It's two small local artifacts. Nothing phones home.

We chose `intfloat/multilingual-e5-small` as the default brain: ~118M parameters, 384 dimensions, MIT-licensed (commercially safe for a bank), and small enough to have a real shot at the CPU budget. The encoder's name is a config value, not a hardcoded import — so swapping brains is a one-line change, never a code change.

---

## Two ways to draw the regions: Approach A and Approach B

We built two versions of the system that share ~90% of their code and have an **identical inference path**. They differ only in *offline training*.

### Approach A — Frozen embeddings + Logistic Regression

The purest form. Take the encoder *exactly as it shipped*, embed every labeled example once, and fit a logistic-regression head with `class_weight="balanced"` (more on why that matters in a second). That's it. It's the fastest thing to ship and the honest baseline — the floor that anything fancier has to beat.

The catch: a generic encoder wasn't trained to care about *your* bank's hair-splitting distinctions. In its map, "card declined" and "card not working" might sit almost on top of each other, because in general-purpose language they basically *are* the same. Approach A inherits whatever separation the pretrained encoder happened to give you.

### Approach B — SetFit: teaching the map to separate look-alikes

This is the upgrade, and it's the answer to the confusable-intents problem.

**SetFit** fine-tunes the encoder *contrastively*. Instead of training it to predict labels, you show it **pairs**: "these two sentences are the same intent — pull them closer together," "these two are different intents — push them apart." Do that across thousands of pairs and the map physically rearranges: the "declined" cluster and the "not working" cluster drift apart, opening a gap the logistic head can cleanly slice through.

Two things make SetFit the right tool here specifically:

1. **It's built for the few-shot, imbalanced regime — which is exactly our data.** Our real dataset is ~1,500 examples across 17 intents, and the imbalance is brutal: some intents have 243 examples, some have **8**. A conventional classifier basically ignores a class with 8 examples. SetFit's contrastive pairing squeezes enormous signal out of those 8 — every example pairs against many others, so eight examples become dozens of training *pairs*. The tiny tail is precisely where SetFit earns its keep.
2. **It costs nothing at inference.** The fine-tuning happens *offline*. Afterward, the fine-tuned encoder is just… an encoder. One forward pass, same as Approach A. You pay for the better map once, at training time, and run it for free forever.

The strategy is: **ship A as the baseline, add B as the contender, and let the data referee.** Which brings us to the single most important artifact in the whole repo.

---

## The confusion matrix is the compass

If you take one practical thing from this project, take this: **the confusion matrix is not a report you generate at the end. It's the steering wheel.**

A confusion matrix shows, for every true intent, what the model *actually predicted*. The diagonal is where it's right. The off-diagonal cells are where it's confused — and each one is a specific, actionable story:

- A big off-diagonal cell between `card_declined` and `card_blocked`? Those two intents are colliding. Either Approach B needs to pull them apart, or — uncomfortable truth — **maybe they're the same intent** and the taxonomy is wrong. The matrix surfaces duplicate intents that humans argued into existence in a spec meeting.
- A whole row that's smeared across many columns? That intent is under-defined or starved of data.

So `evaluate.py` doesn't just print accuracy. It prints **macro-F1** (the headline number — it weights all 17 intents equally, so the 8-example class counts as much as the 243-example one, which plain accuracy would happily ignore), **per-intent F1** (so you see *exactly* which intents are failing), and it writes the full confusion matrix to CSV plus a ranked list of the worst confused pairs. When you compare Approach A vs B, you don't ask "did macro-F1 go up?" — you ask "did the *specific cell* between our two worst look-alikes shrink?"

That's the loop: train → read the matrix → fix the worst collision → retrain → read the matrix again.

---

## Knowing when you don't know

Here's a subtle failure mode that separates toy classifiers from production ones. A customer types something completely outside the 17 intents — *"what's the weather"*, or a furious rant, or pure gibberish. A naive classifier will confidently shout the *nearest* intent, because softmax always sums to 1; it has no vocabulary for "none of these."

That's dangerous in banking. Confidently routing an off-topic message to "wire transfer" is worse than admitting confusion.

So the design always reserves an explicit **fallback / out-of-scope (OOS)** outcome, with three planned strategies (config-selectable):

- **Threshold** — if the top probability is below a calibrated cutoff, it's fallback. Simple, but softmax is overconfident, so this is the crude baseline.
- **Prototype distance** — keep a centroid (a prototype point) for each intent; if a message lands too far from *every* centroid, it's nobody's intent. This fits the embedding-space worldview beautifully, and as a bonus it's how you add a new intent instantly: a new intent is just a new centroid.
- **Energy** — a more calibrated score over the raw logits, stronger than naked softmax.

The cutoff isn't guessed — it's **calibrated on a dev set** that includes real out-of-scope examples (this is what the public CLINC150 dataset, which ships labeled OOS examples, is for). And OOS recall/precision get reported *separately* from in-scope accuracy, because they're a different promise: "when the user is off-topic, how often do we correctly say so?"

*(Status note: this gate is designed and specced but not yet wired into the running code — it's the next major piece.)*

---

## Where the 100 milliseconds actually go

The latency budget is a *systems* problem, not a model problem, and it's worth seeing where the time goes. The plan attacks it in order of leverage:

1. **Export the encoder to ONNX Runtime.** PyTorch is built for flexibility and training; ONNX Runtime is built to execute a frozen graph fast on CPU. Same math, fewer overheads.
2. **Quantize to int8.** Store and compute the encoder's weights as 8-bit integers instead of 32-bit floats. ~4× smaller, and *substantially* faster — **on the right hardware.**
3. **Truncate the input.** We only feed the last 1–2 conversation turns plus the current message, capped at ~96 tokens. This is the cheapest lever of all: transformer cost grows with sequence length, and research consistently shows that for intent classification, *more conversation history actively hurts* — it adds noise and latency for no accuracy. Less context is both faster and better.
4. **Tune for batch-of-one.** Production is single requests, so threads are set for one-at-a-time latency, not throughput, and padding is skipped.

And here's the trap that's loud enough to deserve its own paragraph, written in capitals in the project's constraints:

> **Do not trust latency numbers measured on the development Mac.**

Int8's big speedup leans on a specific CPU feature — **AVX-512-VNNI** — that x86 server chips have and Apple Silicon does not. On a VNNI machine, int8 might be ~2.5× faster; without it, maybe 1.25×. So a number measured on the dev laptop could be off by 2× from production, in either direction. The discipline: **benchmark only on production-equivalent x86 VNNI hardware**, standardize on ONNX Runtime's CPU provider as the single source of truth, and gate the build with a regression test that *fails* if p95 creeps over budget. The dev machine is for writing code, never for trusting milliseconds.

*(Status note: the ONNX/int8 path and the latency harness are specced but not yet built — no latency has been validated yet. The architecture is shaped to hit the budget; the proof is pending.)*

---

## The Hinglish problem

English is easy. Devanagari Hindi is handled by picking a genuinely multilingual encoder — the frozen brain already speaks it.

**Hinglish is the boss fight.** *"mujhe apna balance check karna hai"* is romanized Hindi with English words mixed in. It's wildly common in Indian messaging and almost absent from public training sets. Two defenses:

- **Optional transliteration** — romanized Hindi → Devanagari before encoding, so *"mera card"* and *"मेरा कार्ड"* hit the encoder as the same thing. It's behind a config flag because it's a tradeoff, not a free win, and needs to be measured.
- **SetFit's few-shot strength again** — bootstrap from a *handful* of real Hinglish examples per intent, and let contrastive training generalize from them. This is the same property that rescues the 8-example tail, pointed at a different problem.

The non-negotiable rule baked into the project: **never drop Hindi/Hinglish handling to chase a prettier English number.** It's easy to over-optimize for the majority language and quietly fail the people typing in the other two.

*(Status note: transliteration and language-ID exist as config-gated stubs today; the Hinglish evaluation is future work.)*

---

## Adding an intent on a Tuesday, without a retrain

Step back and look at the operational payoff, because this is *why* the architecture is shaped the way it is.

A product manager says: "We're launching a new card-dispute flow Thursday. We need a `dispute_transaction` intent." In a monolithic fine-tuned system, that's a retraining job, a validation cycle, a deploy — days.

Here, it's:

1. Drop a file of example utterances into `data/intents/dispute_transaction.txt`.
2. Refit the logistic head (seconds) — or, even cheaper, add one prototype centroid for the new intent.
3. Re-run `evaluate.py` and confirm two things: the new intent has decent F1, **and the old intents didn't regress** (a "catastrophic forgetting" check).

The encoder — the expensive, slow, frozen brain — is never touched. *That* is what "decouple understanding from decision" buys you in business terms: the system keeps pace with a product roadmap instead of bottlenecking it.

*(Status note: the `add_intent.py` script is specced; the architecture fully supports it today via a head refit.)*

---

## A quiet design philosophy: config over code

One last thing, because it shapes how the whole repo feels to work in: **there are no magic numbers in the code.** Sequence length, history turns, the OOS threshold, the model name, the SetFit hyperparameters, thread counts, file paths — every tunable lives in a single `config/config.yaml`. Code reads config; command-line flags only *override* it. Seeds are fixed at 42 everywhere, so runs reproduce.

Why care? Because this system is meant to be *operated*, not just *built* — tuned weekly by people who shouldn't have to read Python to change a threshold. And because the two approaches (frozen vs SetFit) and the two backends (PyTorch vs ONNX/int8) are meant to be swapped by flipping a config value, not by editing the hot path. The modules are small and single-purpose — `preprocess`, `encoder`, `classifier`, `oos`, `pipeline` — each doing one thing, so the backend can change underneath a stable interface.

It's not glamorous. It's the difference between a research notebook and something a team can actually run.

---

## Where it stands, honestly

What's real and working today: both brains (frozen **and** SetFit-fine-tuned), the data prep with a safeguard for those 8-example classes, training, and the full evaluation harness with the all-important confusion matrix. It's been proven end-to-end on synthetic data — the wiring is solid.

What's designed but not yet built: the out-of-scope gate, the ONNX/int8 latency engineering (so *no* latency claim is validated yet), the serving API, the one-command intent-adder, and the Hinglish evaluation. The honest status of each is tracked in the README so nobody mistakes a plan for a measurement.

---

## The takeaway

The whole project rests on a single inversion of intuition. Faced with a hard language problem under brutal constraints, the instinct is to reach for a *bigger* model that does *more*. The winning move was the opposite: **freeze a modest model so it does one thing — understand — and put all the fast, cheap, frequently-changing intelligence in a featherweight head that does the other thing — decide.**

Understanding is hard and slow and shared. Deciding is easy and fast and yours to change. Keep them apart, and a 100 ms, on-prem, multilingual, ever-growing banking classifier stops being a contradiction and starts being a config file.

---

*For the operational details — exact commands, file layout, dependencies, and current build status — see [`README.md`](../README.md). For the non-negotiable design constraints, see [`CLAUDE.md`](../CLAUDE.md).*
