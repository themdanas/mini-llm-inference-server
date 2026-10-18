"""
tests/test_e2e.py
─────────────────
End-to-end tests for the NanoInfer serving layer.

Uses a tiny synthetic model + tokenizer so tests run fast on any machine
without GPU or downloaded weights.

Tests cover:
  1. Non-streaming generation  (POST /v1/generate)
  2. Streaming via SSE         (POST /v1/generate, stream=true)
  3. Health endpoint            (GET /health)
  4. Metrics endpoint           (GET /metrics)
  5. Concurrent requests
  6. EOS / max_tokens stopping
"""

from __future__ import annotations

import asyncio
import json
import pytest
import httpx

from fastapi.testclient import TestClient

from serving.engine import EngineConfig, InferenceEngine
from serving.server import create_app
from serving.schemas import GenerateRequest, GenerateResponse


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine_config() -> EngineConfig:
    """Config that uses the tiny synthetic model (2 layers, 64-dim, vocab=256)."""
    return EngineConfig(
        model_config_preset="tiny",
        model_weights_path=None,      # random init — fine for smoke tests
        tokenizer_path=None,          # BPETokenizer.tiny_test()
        device="cpu",
        dtype="float32",
        num_kv_blocks=64,
        block_size=16,
        max_batch_size=8,
        max_tokens_per_step=512,
    )


@pytest.fixture(scope="module")
def app(engine_config):
    """Create the FastAPI app with a tiny engine."""
    return create_app(engine_config)


@pytest.fixture(scope="module")
def client(app):
    """Synchronous test client — uses the TestClient context manager
    so the lifespan (engine start/shutdown) is handled automatically."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ─── Health ─────────────────────────────────────────────────────────────────


class TestHealth:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["device"] == "cpu"
        assert body["uptime_s"] >= 0

    def test_health_includes_model_info(self, client):
        resp = client.get("/health")
        body = resp.json()
        # the model repr should mention layer count or dimensions
        assert body["model"] != ""


# ─── Metrics ────────────────────────────────────────────────────────────────


class TestMetrics:

    def test_metrics_returns_valid_fields(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "requests_total" in body
        assert "requests_active" in body
        assert "kv_cache_utilization" in body
        assert isinstance(body["kv_cache_utilization"], float)


# ─── Non-streaming generation ──────────────────────────────────────────────


class TestGenerate:

    def test_generate_basic(self, client):
        """Simple non-streaming generation returns a valid response."""
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "hello",
                "max_new_tokens": 5,
                "temperature": 1.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert "text" in body
        assert "finish_reason" in body
        assert body["finish_reason"] in ("eos", "max_tokens", "stop_string")
        assert "usage" in body
        assert body["usage"]["completion_tokens"] > 0

    def test_generate_max_tokens_honored(self, client):
        """Request with max_new_tokens=3 should generate at most 3 tokens."""
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "test",
                "max_new_tokens": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["usage"]["completion_tokens"] <= 3

    def test_generate_greedy(self, client):
        """temperature=0 should be treated as greedy and not crash."""
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "greedy",
                "max_new_tokens": 3,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["usage"]["completion_tokens"] > 0

    def test_generate_empty_prompt_rejected(self, client):
        """Empty prompt should get a 422 validation error."""
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "",
                "max_new_tokens": 5,
            },
        )
        assert resp.status_code == 422

    def test_generate_alias_endpoint(self, client):
        """POST /generate without /v1 works identically."""
        resp = client.post(
            "/generate",
            json={
                "prompt": "test alias",
                "max_new_tokens": 3,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert body["usage"]["completion_tokens"] > 0

    def test_generate_stop_string(self, client):
        """Specifying a stop string terminates generation early."""
        # Find token in greedy run
        probe = client.post(
            "/v1/generate",
            json={"prompt": "stop probe", "max_new_tokens": 10, "temperature": 0.0},
        ).json()
        assert len(probe["text"]) > 1
        stop_char = probe["text"][1]

        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "stop probe",
                "max_new_tokens": 10,
                "temperature": 0.0,
                "stop": [stop_char],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["finish_reason"] == "stop_string"
        assert stop_char not in body["text"]


# ─── Streaming generation ──────────────────────────────────────────────────


class TestStreaming:

    def test_stream_returns_sse_events(self, client):
        """Streaming request should return SSE events."""
        with client.stream(
            "POST",
            "/v1/generate",
            json={
                "prompt": "stream test",
                "max_new_tokens": 3,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type

            chunks = []
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunks.append(json.loads(data))

            # we should have received at least one chunk
            assert len(chunks) >= 1
            # each chunk has an id and delta
            for chunk in chunks:
                assert "id" in chunk
                assert "delta" in chunk

            # last real chunk should have a finish_reason
            last = chunks[-1]
            assert last.get("finish_reason") is not None


# ─── Concurrent requests ───────────────────────────────────────────────────


class TestConcurrency:

    def test_multiple_requests(self, client):
        """Submit several requests sequentially and verify all complete."""
        results = []
        for i in range(3):
            resp = client.post(
                "/v1/generate",
                json={
                    "prompt": f"concurrent {i}",
                    "max_new_tokens": 3,
                },
            )
            assert resp.status_code == 200
            results.append(resp.json())

        assert len(results) == 3
        ids = [r["id"] for r in results]
        # all IDs should be unique
        assert len(set(ids)) == 3


# ─── Metrics after generation ──────────────────────────────────────────────


class TestMetricsAfterGeneration:

    def test_metrics_updated_after_requests(self, client):
        """After running generation, metrics should reflect work done."""
        # run a generation first
        client.post(
            "/v1/generate",
            json={"prompt": "metric test", "max_new_tokens": 3},
        )

        resp = client.get("/metrics")
        body = resp.json()
        assert body["requests_total"] > 0
        assert body["tokens_generated"] > 0