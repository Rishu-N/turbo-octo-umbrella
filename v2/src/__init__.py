"""Self-improving semantic-cache intent classifier (v2).

A new query is classified by looking it up against a growing cache of already-labeled
queries (exact-match -> semantic NN over Jina v3 embeddings) instead of always calling
an expensive LLM. Misses fall through to an LLM classifier whose answer is written back
into the cache (behind a confidence gate), so hit-rate rises over time without retraining.

See PROJECT_CONTEXT.md for the full design and CLAUDE.md for the hard constraints.
"""
