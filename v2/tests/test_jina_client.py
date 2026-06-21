"""Phase 2 tests: the isolated Jina v3 client (src/embeddings/jina_client.py).

The HTTP layer is faked via an injected ``session``, so these run fully offline — no network, no key.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import requests

from src.embeddings.jina_client import JinaAPIError, JinaClient


def _cfg(tmp_path, **over) -> dict:
    jina = {
        "api_url": "https://example.test/v1/embeddings",
        "model": "jina-embeddings-v3",
        "task": "classification",
        "dimensions": 8,
        "normalize": True,
        "timeout_seconds": 1.0,
        "max_retries": 2,
        "backoff_base_seconds": 0.0,
        "batch_size": 2,
        "embed_cache_path": str(tmp_path / "embed_cache.sqlite"),
    }
    jina.update(over)
    return {"jina": jina}


def _emb_for(text: str, dim: int) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed).standard_normal(dim).tolist()


class _Resp:
    def __init__(self, embeddings, status=200):
        self._emb = embeddings
        self.status_code = status

    def json(self):
        return {"data": [{"embedding": e, "index": i} for i, e in enumerate(self._emb)], "usage": {}}


class FakeSession:
    """Returns deterministic per-text embeddings; records call count and payloads."""

    def __init__(self, dim: int):
        self.dim = dim
        self.calls = 0
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        self.payloads.append(json)
        return _Resp([_emb_for(t, self.dim) for t in json["input"]])


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("src.embeddings.jina_client.time.sleep", lambda *_: None)


@pytest.fixture
def _key(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-key")


def test_embed_shape_and_l2_normalized(tmp_path, _key):
    c = JinaClient(_cfg(tmp_path), session=FakeSession(8))
    vecs = c.embed(["a", "b", "c"])
    assert vecs.shape == (3, 8)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_request_payload_uses_classification_adapter(tmp_path, _key):
    sess = FakeSession(8)
    JinaClient(_cfg(tmp_path), session=sess).embed_one("x")
    p = sess.payloads[0]
    assert p["model"] == "jina-embeddings-v3"
    assert p["task"] == "classification"
    assert p["dimensions"] == 8
    assert p["input"] == ["x"]


def test_cache_avoids_repeat_api_calls(tmp_path, _key):
    sess = FakeSession(8)
    c = JinaClient(_cfg(tmp_path), session=sess)
    c.embed(["hello"])
    after_first = sess.calls
    again = c.embed_one("hello")
    assert sess.calls == after_first  # served from cache
    assert again.shape == (8,)


def test_persistent_cache_across_instances(tmp_path, _key):
    cfg = _cfg(tmp_path)
    v1 = JinaClient(cfg, session=FakeSession(8)).embed_one("persist")
    fresh = FakeSession(8)
    v2 = JinaClient(cfg, session=fresh).embed_one("persist")
    assert fresh.calls == 0  # second instance reads the on-disk cache
    assert np.allclose(v1, v2)


def test_batching_respects_batch_size(tmp_path, _key):
    sess = FakeSession(8)
    JinaClient(_cfg(tmp_path, batch_size=2), session=sess).embed(["a", "b", "c", "d", "e"])
    assert sess.calls == 3  # ceil(5 / 2)
    assert all(len(p["input"]) <= 2 for p in sess.payloads)


def test_dedupe_within_single_call(tmp_path, _key):
    sess = FakeSession(8)
    out = JinaClient(_cfg(tmp_path, batch_size=10), session=sess).embed(["same", "same", "same"])
    assert out.shape == (3, 8)
    assert sum(len(p["input"]) for p in sess.payloads) == 1  # only one unique text requested
    assert np.allclose(out[0], out[1]) and np.allclose(out[1], out[2])


def test_empty_input(tmp_path, _key):
    out = JinaClient(_cfg(tmp_path), session=FakeSession(8)).embed([])
    assert out.shape == (0, 8)


# --- resilience --------------------------------------------------------------------------------
class _FlakySession(FakeSession):
    def __init__(self, dim, fail_times):
        super().__init__(dim)
        self.fail_times = fail_times

    def post(self, url, json=None, headers=None, timeout=None):
        if self.calls < self.fail_times:
            self.calls += 1
            raise requests.exceptions.ConnectionError("boom")
        return super().post(url, json=json, headers=headers, timeout=timeout)


class _StatusSession(FakeSession):
    def __init__(self, dim, statuses):
        super().__init__(dim)
        self.statuses = list(statuses)

    def post(self, url, json=None, headers=None, timeout=None):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        self.payloads.append(json)
        if status == 200:
            return _Resp([_emb_for(t, self.dim) for t in json["input"]], 200)
        return _Resp([], status)


def test_retry_on_connection_error_then_success(tmp_path, _key):
    sess = _FlakySession(8, fail_times=2)
    v = JinaClient(_cfg(tmp_path, max_retries=3), session=sess).embed_one("retry")
    assert v.shape == (8,)


def test_server_error_5xx_is_retried(tmp_path, _key):
    sess = _StatusSession(8, [500, 200])
    v = JinaClient(_cfg(tmp_path, max_retries=3), session=sess).embed_one("x")
    assert v.shape == (8,) and sess.calls == 2


def test_client_error_4xx_is_fatal_not_retried(tmp_path, _key):
    sess = _StatusSession(8, [400])
    with pytest.raises(JinaAPIError):
        JinaClient(_cfg(tmp_path, max_retries=3), session=sess).embed_one("x")
    assert sess.calls == 1  # not retried


def test_exhausted_retries_raise(tmp_path, _key):
    class _Dead:
        def __init__(self):
            self.calls = 0

        def post(self, *a, **k):
            self.calls += 1
            raise requests.exceptions.ConnectionError("down")

    with pytest.raises(JinaAPIError):
        JinaClient(_cfg(tmp_path, max_retries=2), session=_Dead()).embed_one("x")


def test_missing_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with pytest.raises(JinaAPIError):
        JinaClient(_cfg(tmp_path), session=FakeSession(8)).embed_one("x")
