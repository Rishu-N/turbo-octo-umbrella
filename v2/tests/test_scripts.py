"""Phase 6 tests: the calibration / eval / simulation / audit / seeding script logic (offline)."""

from __future__ import annotations

from src.config import load_config
from src.embeddings.mock import MockEmbedder
from src.llm.fallback import StubLLMClassifier
from src.pipeline import Pipeline
from src.preprocess.normalize import normalize
from src.store.exact_store import ExactStore
from src.store.vector_store import VectorStore

from scripts.audit_cache import summarize
from scripts.calibrate_thresholds import collect_scores, recommend, sweep_thresholds
from scripts.evaluate import evaluate_pipeline
from scripts.seed_cache import seed_stores
from scripts.simulate_growth import make_stream, simulate, sweep_t_write


def _cfg(tmp_path) -> dict:
    cfg = load_config()
    cfg["exact_store"]["path"] = str(tmp_path / "exact.sqlite")
    cfg["vector_store"]["index_path"] = str(tmp_path / "vs.bin")
    cfg["vector_store"]["meta_path"] = str(tmp_path / "meta.sqlite")
    cfg["audit"]["log_path"] = str(tmp_path / "audit.jsonl")
    return cfg


# --- calibrate_thresholds: pure sweep / recommend ----------------------------------------------
def test_sweep_thresholds_shape_and_monotonic_hit_rate():
    scores = [(0.90, "a", "a"), (0.80, "a", "b"), (0.70, "c", "c"), (0.60, "d", "e")]
    curve = sweep_thresholds(scores, [0.65, 0.75, 0.85, 0.95])
    # hit-rate is non-increasing as the threshold rises
    hit_rates = [row["hit_rate"] for row in curve]
    assert hit_rates == sorted(hit_rates, reverse=True)
    assert curve[0]["hit_rate"] == 0.75 and curve[-1]["hit_rate"] == 0.0  # >=0.65 keeps 3/4; >=0.95 keeps 0


def test_recommend_picks_low_threshold_within_false_hit_budget():
    scores = [(0.95, "a", "a"), (0.90, "a", "a"), (0.70, "x", "y"), (0.60, "p", "q")]
    curve = sweep_thresholds(scores, [0.65, 0.85, 0.92, 0.97])
    # at 0.85 only the two correct (>=0.9) remain -> false_hit 0; that's the max-coverage safe threshold
    assert recommend(curve, max_false_hit=0.0) == 0.85


def test_collect_scores_finds_seeded_match(tmp_path):
    cfg = _cfg(tmp_path)
    emb = MockEmbedder(cfg["vector_store"]["dim"])
    vec = VectorStore(cfg)
    vec.add(emb.embed_one(normalize("what is my balance", cfg)), "balance_inquiry", "seed")
    scores = collect_scores(cfg, emb, vec, [{"query": "what is my balance", "intent": "balance_inquiry"}], namespaced=False)
    assert len(scores) == 1
    sim, pred, gold = scores[0]
    assert sim > 0.99 and pred == "balance_inquiry" and gold == "balance_inquiry"


# --- seed_cache --------------------------------------------------------------------------------
def test_seed_stores_writes_both_stores(tmp_path):
    cfg = _cfg(tmp_path)
    emb = MockEmbedder(cfg["vector_store"]["dim"])
    exact, vector = ExactStore(cfg), VectorStore(cfg)
    rows = [
        {"query": "what is my balance", "intent": "balance_inquiry", "lang": "en", "prev_intent": ""},
        {"query": "yes do it", "intent": "card_lost_or_stolen", "lang": "en", "prev_intent": "card_lost_or_stolen"},
    ]
    summary = seed_stores(cfg, emb, exact, vector, rows)
    assert summary["rows"] == 2 and summary["vectors"] == 2
    assert exact.get(normalize("what is my balance", cfg)).intent == "balance_inquiry"
    # the referential row is namespaced, not in the query-only bucket
    assert exact.get(normalize("yes do it", cfg)) is None
    assert exact.get(normalize("yes do it", cfg), prev_intent="card_lost_or_stolen") is not None


# --- evaluate ----------------------------------------------------------------------------------
def test_evaluate_pipeline_reports_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    emb = MockEmbedder(cfg["vector_store"]["dim"])
    p = Pipeline(cfg, jina=emb, llm=StubLLMClassifier(cfg))
    p._exact.put(normalize("what is my balance", cfg), "balance_inquiry", "seed")
    rows = [
        {"query": "what is my balance", "intent": "balance_inquiry", "lang": "en", "prev_intent": ""},
        {"query": "i lost my card", "intent": "card_lost_or_stolen", "lang": "en", "prev_intent": ""},
    ]
    m = evaluate_pipeline(cfg, p, rows)
    for key in ("accuracy", "macro_f1", "hit_rate", "false_hit_rate", "by_source", "per_language_accuracy", "oos"):
        assert key in m
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["by_source"].get("exact_cache", 0) >= 1  # the seeded query is an exact hit


# --- simulate_growth ---------------------------------------------------------------------------
def test_simulation_grows_cache_and_runs(tmp_path):
    cfg = _cfg(tmp_path)
    emb = MockEmbedder(cfg["vector_store"]["dim"])
    rows = [
        {"query": f"q{i}", "intent": ["balance_inquiry", "transfer_funds", "loan_inquiry"][i % 3], "prev_intent": ""}
        for i in range(9)
    ]
    stream = make_stream(rows, repeats=6, seed=1)
    res = simulate(cfg, emb, stream, t_write=0.5, noise=0.2, seed=1)
    assert res["n"] == len(stream)
    assert res["hit_rate"] > 0.0  # repeats -> exact-cache hits accrue
    assert len(res["hit_rate_series"]) == len(stream)


def test_higher_t_write_reduces_caching_and_errors(tmp_path):
    cfg = _cfg(tmp_path)
    emb = MockEmbedder(cfg["vector_store"]["dim"])
    rows = [
        {"query": f"q{i}", "intent": ["balance_inquiry", "transfer_funds", "loan_inquiry"][i % 3], "prev_intent": ""}
        for i in range(9)
    ]
    stream = make_stream(rows, repeats=6, seed=2)
    low, high = sweep_t_write(cfg, emb, stream, [0.50, 0.95], noise=0.3, seed=2)
    # A stricter write-back gate caches fewer answers and admits fewer cached errors.
    assert low["hit_rate"] >= high["hit_rate"]
    assert low["cache_false_hit_rate"] >= high["cache_false_hit_rate"]


# --- audit_cache -------------------------------------------------------------------------------
def test_audit_summary_and_purge(tmp_path):
    cfg = _cfg(tmp_path)
    exact = ExactStore(cfg)
    exact.put("a", "balance_inquiry", "seed")
    exact.put("b", "transfer_funds", "writeback", 0.95)
    exact.put("b", "loan_inquiry", "writeback", 0.95)  # conflict (not overwritten)
    summary = summarize(exact)
    assert summary["total"] == 2 and summary["by_source"]["seed"] == 1
    assert summary["conflicts"] == 1 and len(summary["conflict_details"]) == 1
    removed = exact.purge_writebacks()
    assert removed == 1
    assert exact.get("a") is not None and exact.get("b") is None  # seed kept, writeback purged
