"""Phase 6 test: the optional FastAPI endpoint (src/serve.py). Skipped if fastapi isn't installed."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # starlette's TestClient needs httpx

from fastapi.testclient import TestClient  # noqa: E402

from src.pipeline import Classification  # noqa: E402
from src.serve import create_app  # noqa: E402


class _FakePipeline:
    def classify(self, query, history=None, previous_intent=None):
        return Classification("balance_inquiry", 0.91, "exact_cache", False)


def test_classify_endpoint_returns_expected_shape():
    app = create_app(cfg={}, pipeline=_FakePipeline())
    client = TestClient(app)
    resp = client.post("/classify", json={"query": "what is my balance?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "intent": "balance_inquiry",
        "confidence": 0.91,
        "source": "exact_cache",
        "is_oos": False,
    }


def test_classify_endpoint_accepts_history_and_previous_intent():
    app = create_app(cfg={}, pipeline=_FakePipeline())
    client = TestClient(app)
    resp = client.post(
        "/classify",
        json={"query": "yes do it", "history": ["i lost my card"], "previous_intent": "card_lost_or_stolen"},
    )
    assert resp.status_code == 200 and resp.json()["source"] == "exact_cache"
