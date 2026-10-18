"""
tests/test_serving.py
─────────────────────
Comprehensive unit and integration test suite for the NanoInfer serving layer:
  - Schemas (validation, constraints, serialization)
  - Streaming (RequestStream, make_sse_response, cancellation)
  - InferenceEngine (lifecycle, generation, greedy, stop strings, presets, metrics, leak checks)
  - Routes & Server (FastAPI endpoints, aliases, 422 errors, 503 fallback, CLI arg parsing)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from serving.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    StreamChunk,
    UsageInfo,
    ErrorResponse,
)
from serving.streaming import RequestStream, make_sse_response
from serving.engine import EngineConfig, InferenceEngine
from serving.server import create_app, parse_args
from model.config import ModelConfig


# ─── 1. Schemas Tests ────────────────────────────────────────────────────────


class TestServingSchemas:
    """Validate request and response schemas and constraint enforcements."""

    def test_generate_request_defaults(self):
        req = GenerateRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.max_new_tokens == 256
        assert req.temperature == 1.0
        assert req.top_k == 0
        assert req.top_p == 1.0
        assert req.min_p == 0.0
        assert req.repetition_penalty == 1.0
        assert req.stop == []
        assert req.stop_token_ids == []
        assert req.stream is False
        assert req.priority == 0

    def test_generate_request_custom_valid(self):
        req = GenerateRequest(
            prompt="write code",
            max_new_tokens=100,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.1,
            stop=["\n", "END"],
            stop_token_ids=[2],
            stream=True,
            priority=5,
        )
        assert req.max_new_tokens == 100
        assert req.stream is True
        assert req.stop == ["\n", "END"]

    def test_generate_request_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_generate_request_invalid_temperature(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", temperature=-0.5)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", temperature=101.0)

    def test_generate_request_invalid_top_p(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", top_p=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", top_p=1.1)

    def test_generate_request_invalid_min_p(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", min_p=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", min_p=1.1)

    def test_generate_request_invalid_repetition_penalty(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", repetition_penalty=0.9)

    def test_generate_request_invalid_max_tokens(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", max_new_tokens=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", max_new_tokens=8193)

    def test_generate_response_serialization(self):
        usage = UsageInfo(prompt_tokens=4, completion_tokens=10, total_tokens=14)
        resp = GenerateResponse(
            id="req-123",
            text="world",
            finish_reason="eos",
            usage=usage,
        )
        d = resp.model_dump()
        assert d["id"] == "req-123"
        assert d["text"] == "world"
        assert d["finish_reason"] == "eos"
        assert d["usage"]["total_tokens"] == 14

    def test_stream_chunk_serialization(self):
        chunk = StreamChunk(id="req-1", delta="hel")
        assert chunk.id == "req-1"
        assert chunk.delta == "hel"
        assert chunk.finish_reason is None

        final = StreamChunk(id="req-1", delta="", finish_reason="stop_string")
        assert final.finish_reason == "stop_string"

    def test_health_response_model(self):
        h = HealthResponse(status="ok", model="tiny", device="cpu", uptime_s=12.5)
        assert h.status == "ok"
        assert h.device == "cpu"
        assert h.uptime_s == 12.5

    def test_metrics_response_model(self):
        m = MetricsResponse(
            requests_total=10,
            requests_active=1,
            requests_waiting=2,
            tokens_generated=150,
            avg_ttft_ms=12.3,
            avg_latency_ms=45.6,
            kv_cache_utilization=0.25,
        )
        assert m.requests_total == 10
        assert m.kv_cache_utilization == 0.25

    def test_error_response_model(self):
        err = ErrorResponse(error="Not Found", detail="model missing")
        assert err.error == "Not Found"
        assert err.detail == "model missing"


# ─── 2. Streaming Layer Tests ───────────────────────────────────────────────


class TestRequestStream:
    """Test RequestStream async iterator and sse response wrapper."""

    @pytest.mark.anyio
    async def test_stream_push_and_consume(self):
        stream = RequestStream("req-test")
        c1 = StreamChunk(id="req-test", delta="A")
        c2 = StreamChunk(id="req-test", delta="B")
        c3 = StreamChunk(id="req-test", delta="", finish_reason="eos")

        await stream.push(c1)
        await stream.push(c2)
        await stream.push(c3)
        await stream.finish()

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert len(collected) == 3
        assert [c.delta for c in collected] == ["A", "B", ""]
        assert collected[-1].finish_reason == "eos"

    @pytest.mark.anyio
    async def test_stream_cancelled_drops_pushes(self):
        stream = RequestStream("req-cancelled")
        stream.cancelled = True
        await stream.push(StreamChunk(id="req-cancelled", delta="dropped"))
        await stream.finish()

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert len(collected) == 0

    @pytest.mark.anyio
    async def test_make_sse_response_formatting(self):
        stream = RequestStream("req-sse")
        await stream.push(StreamChunk(id="req-sse", delta="token1"))
        await stream.finish()

        response = make_sse_response(stream)
        assert response.media_type == "text/event-stream"

        items = []
        async for raw in response.body_iterator:
            items.append(raw["data"] if isinstance(raw, dict) else str(raw))

        assert any("token1" in item for item in items)
        assert "[DONE]" in items


# ─── 3. InferenceEngine Tests ───────────────────────────────────────────────


class TestInferenceEngine:
    """Thorough unit tests for InferenceEngine lifecycle, generation, stop logic, and cleanup."""

    @pytest.fixture
    def tiny_config(self) -> EngineConfig:
        return EngineConfig(
            model_config_preset="tiny",
            device="cpu",
            dtype="float32",
            num_kv_blocks=32,
            block_size=16,
            max_batch_size=4,
            max_tokens_per_step=128,
        )

    @pytest.mark.anyio
    async def test_engine_unstarted_raises(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        req = GenerateRequest(prompt="hello")
        with pytest.raises(RuntimeError, match="not started"):
            await engine.generate(req)
        with pytest.raises(RuntimeError, match="not started"):
            await engine.generate_stream(req)

    @pytest.mark.anyio
    async def test_engine_health_before_and_after_start(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        # before start
        h_pre = engine.health()
        assert h_pre.status == "loading"
        assert h_pre.uptime_s == 0.0

        # start
        await engine.start()
        h_post = engine.health()
        assert h_post.status == "ok"
        assert h_post.uptime_s >= 0.0
        assert h_post.device == "cpu"
        assert "ModelConfig" in h_post.model

        # shutdown
        await engine.shutdown()

    @pytest.mark.anyio
    async def test_engine_lifecycle_idempotency(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        await engine.start()
        # double start should be safe
        await engine.start()
        assert engine._running is True

        # double shutdown should be safe
        await engine.shutdown()
        await engine.shutdown()
        assert engine._running is False

    @pytest.mark.anyio
    async def test_engine_load_custom_json_preset(self, tmp_path):
        cfg = ModelConfig.tiny()
        json_path = tmp_path / "custom_cfg.json"
        cfg.to_json(str(json_path))

        engine_cfg = EngineConfig(
            model_config_preset=str(json_path),
            device="cpu",
            num_kv_blocks=32,
        )
        engine = InferenceEngine(engine_cfg)
        await engine.start()
        assert engine._model_config.d_model == cfg.d_model
        assert engine._model_config.n_layers == cfg.n_layers
        await engine.shutdown()

    @pytest.mark.anyio
    async def test_engine_generate_greedy_deterministic(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        await engine.start()

        req1 = GenerateRequest(prompt="test prompt", max_new_tokens=6, temperature=0.0)
        res1 = await engine.generate(req1)

        req2 = GenerateRequest(prompt="test prompt", max_new_tokens=6, temperature=0.0)
        res2 = await engine.generate(req2)

        assert res1.text == res2.text
        assert res1.usage.completion_tokens == 6
        assert res1.finish_reason in ("max_tokens", "eos")

        # Verify no memory leak
        assert len(engine._request_kv_caches) == 0
        assert len(engine._request_streams) == 0
        assert len(engine._request_completions) == 0
        assert len(engine._request_results) == 0

        await engine.shutdown()

    @pytest.mark.anyio
    async def test_engine_generate_stop_string_stops_early(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        await engine.start()

        # Run greedy first to find what token is emitted
        probe_req = GenerateRequest(prompt="probe", max_new_tokens=10, temperature=0.0)
        probe_res = await engine.generate(probe_req)
        assert len(probe_res.text) > 1

        # Use a character that appears after the first character as the stop sequence
        stop_char = probe_res.text[1]
        req = GenerateRequest(
            prompt="probe",
            max_new_tokens=10,
            temperature=0.0,
            stop=[stop_char],
        )
        res = await engine.generate(req)
        assert res.finish_reason == "stop_string"
        # The output text must be truncated before the stop string
        assert stop_char not in res.text

        # Verify cleanup
        assert len(engine._request_kv_caches) == 0
        assert len(engine._request_streams) == 0
        await engine.shutdown()

    @pytest.mark.anyio
    async def test_engine_generate_stream_reconstructs_text(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        await engine.start()

        # Generate stream
        req = GenerateRequest(prompt="stream test", max_new_tokens=5, temperature=0.0, stream=True)
        stream = await engine.generate_stream(req)

        deltas = []
        finish_reason = None
        async for chunk in stream:
            deltas.append(chunk.delta)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        full_stream_text = "".join(deltas)
        assert len(deltas) >= 1
        assert finish_reason in ("max_tokens", "eos", "stop_string")

        # Non-streaming equivalence
        non_stream_req = GenerateRequest(prompt="stream test", max_new_tokens=5, temperature=0.0, stream=False)
        non_stream_res = await engine.generate(non_stream_req)
        assert full_stream_text == non_stream_res.text

        # Verify streaming cleanup (no memory leak)
        assert len(engine._request_kv_caches) == 0
        assert len(engine._request_streams) == 0
        assert len(engine._request_completions) == 0
        assert len(engine._request_results) == 0
        assert len(engine._streaming_requests) == 0

        await engine.shutdown()

    @pytest.mark.anyio
    async def test_engine_streaming_stop_string(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        await engine.start()

        probe_req = GenerateRequest(prompt="test", max_new_tokens=10, temperature=0.0)
        probe_res = await engine.generate(probe_req)
        stop_char = probe_res.text[1]

        req = GenerateRequest(
            prompt="test",
            max_new_tokens=10,
            temperature=0.0,
            stop=[stop_char],
            stream=True,
        )
        stream = await engine.generate_stream(req)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].finish_reason == "stop_string"

        # Verify clean cleanup
        assert len(engine._request_kv_caches) == 0
        assert len(engine._request_streams) == 0
        await engine.shutdown()

    @pytest.mark.anyio
    async def test_engine_cancel_request_cleans_resources(self, tiny_config):
        engine = InferenceEngine(tiny_config)
        await engine.start()

        req = GenerateRequest(prompt="cancel test", max_new_tokens=50, stream=True)
        stream = await engine.generate_stream(req)

        # cancel after reading one chunk
        async for chunk in stream:
            engine.cancel(stream.request_id)
            break

        await asyncio.sleep(0.02)
        assert len(engine._request_kv_caches) == 0
        assert len(engine._request_streams) == 0
        assert len(engine._streaming_requests) == 0

        await engine.shutdown()

    @pytest.mark.anyio
    async def test_stream_retained_in_engine_until_consumed(self, tiny_config):
        """A stream must not be deleted from engine state before the consumer finishes reading."""
        engine = InferenceEngine(tiny_config)
        await engine.start()

        req = GenerateRequest(prompt="slow consumer", max_new_tokens=8, stream=True)
        stream = await engine.generate_stream(req)

        # Allow the background step loop to generate tokens and finalise
        await asyncio.sleep(0.08)

        # The stream MUST still be present in engine state while unconsumed chunks remain
        assert stream.request_id in engine._request_streams

        # Consume the stream to the end
        chunks = [c async for c in stream]
        assert len(chunks) >= 1

        # Now that consumption has completed, the stream is removed
        assert stream.request_id not in engine._request_streams
        assert engine._kv_cache.used_blocks == 0
        await engine.shutdown()

    @pytest.mark.anyio
    async def test_concurrent_streams_independent_cleanup(self, tiny_config):
        """Two concurrent streams complete independently without cross-request corruption."""
        engine = InferenceEngine(tiny_config)
        await engine.start()
        initial_free = engine._kv_cache.free_blocks

        req1 = GenerateRequest(prompt="stream A", max_new_tokens=20, stream=True)
        req2 = GenerateRequest(prompt="stream B", max_new_tokens=20, stream=True)

        s1 = await engine.generate_stream(req1)
        s2 = await engine.generate_stream(req2)

        # Cancel s1 after reading 2 chunks
        count = 0
        async for _ in s1:
            count += 1
            if count == 2:
                s1.cancelled = True
                s1.close()
                break

        # Let s1 cancellation process
        await asyncio.sleep(0.02)
        assert s1.request_id not in engine._request_streams

        # s2 must still be active and unaffected
        assert s2.request_id in engine._request_streams
        chunks2 = [c async for c in s2]
        assert len(chunks2) >= 1

        # Both streams finished; verify clean state and KV block restoration
        assert len(engine._request_streams) == 0
        assert len(engine._request_kv_caches) == 0
        assert engine._kv_cache.used_blocks == 0
        assert engine._kv_cache.free_blocks == initial_free

        await engine.shutdown()

    @pytest.mark.anyio
    async def test_client_disconnect_during_streaming(self, tiny_config):
        """Simulate client disconnect; ensure scheduler cancels and frees all blocks."""
        engine = InferenceEngine(tiny_config)
        await engine.start()
        initial_free = engine._kv_cache.free_blocks

        req = GenerateRequest(prompt="disconnect midstream", max_new_tokens=40, stream=True)
        stream = await engine.generate_stream(req)

        # Read 1 chunk then disconnect
        async for _ in stream:
            stream.cancelled = True
            stream.close()
            break

        await asyncio.sleep(0.05)
        # Request should be marked finished in scheduler and cleaned in engine
        sched_req = engine._scheduler.get_request(stream.request_id)
        assert sched_req is not None
        assert sched_req.is_finished

        # Blocks returned to allocator
        assert engine._kv_cache.used_blocks == 0
        assert engine._kv_cache.free_blocks == initial_free
        assert engine.metrics().kv_cache_utilization == 0.0
        assert len(engine._request_streams) == 0

        await engine.shutdown()

    @pytest.mark.anyio
    async def test_kv_cache_accounting_exact_allocator(self, tiny_config):
        """Verify scheduler allocator block count returns to baseline across request types."""
        engine = InferenceEngine(tiny_config)
        await engine.start()
        allocator = engine._scheduler._allocator
        initial_free = allocator.num_free_blocks

        # 1. Non-streaming request
        res = await engine.generate(GenerateRequest(prompt="baseline non-stream", max_new_tokens=4))
        assert res.usage.completion_tokens > 0
        assert allocator.num_free_blocks == initial_free
        assert allocator.num_used_blocks == 0

        # 2. Streaming request
        s = await engine.generate_stream(GenerateRequest(prompt="baseline stream", max_new_tokens=4, stream=True))
        _ = [c async for c in s]
        assert allocator.num_free_blocks == initial_free
        assert allocator.num_used_blocks == 0

        # 3. Disconnected stream
        s_disc = await engine.generate_stream(GenerateRequest(prompt="baseline disc", max_new_tokens=30, stream=True))
        async for _ in s_disc:
            s_disc.cancelled = True
            s_disc.close()
            break
        await asyncio.sleep(0.03)
        assert allocator.num_free_blocks == initial_free
        assert allocator.num_used_blocks == 0
        assert engine.metrics().kv_cache_utilization == 0.0

        await engine.shutdown()

    @pytest.mark.anyio
    async def test_generation_failure_after_admission_reclaims_resources(self, tiny_config):
        """When forward/sampling fails during generation, blocks and request state are reclaimed."""
        import unittest.mock as mock
        engine = InferenceEngine(tiny_config)
        await engine.start()
        allocator = engine._scheduler._allocator
        initial_free = allocator.num_free_blocks

        # Submit request
        req = GenerateRequest(prompt="will fail", max_new_tokens=5, stream=True)
        stream = await engine.generate_stream(req)

        # Mock model.forward to raise an error
        with mock.patch.object(engine._model, "forward", side_effect=RuntimeError("Simulated CUDA/forward error")):
            # Step the engine to trigger the failure
            await engine._run_one_step()

        # The failing request must be cancelled and its blocks freed
        sched_req = engine._scheduler.get_request(stream.request_id)
        assert sched_req is not None
        assert sched_req.is_finished
        assert allocator.num_free_blocks == initial_free
        assert allocator.num_used_blocks == 0
        assert len(engine._request_kv_caches) == 0

        await engine.shutdown()


# ─── 4. Routes and Server Tests ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def app():
    cfg = EngineConfig(
        model_config_preset="tiny",
        device="cpu",
        num_kv_blocks=32,
        block_size=16,
    )
    return create_app(cfg)


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestServingRoutesAndServer:
    """Test FastAPI application routes, endpoints, error responses, and CLI args."""

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["device"] == "cpu"
        assert body["uptime_s"] >= 0

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "requests_total" in body
        assert "tokens_generated" in body
        assert "kv_cache_utilization" in body

    def test_generate_v1_endpoint(self, client):
        resp = client.post(
            "/v1/generate",
            json={"prompt": "hello", "max_new_tokens": 4, "temperature": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "text" in data
        assert data["usage"]["completion_tokens"] == 4

    def test_generate_endpoint_alias(self, client):
        """Verify POST /generate works identically to /v1/generate."""
        resp = client.post(
            "/generate",
            json={"prompt": "hello alias", "max_new_tokens": 3, "temperature": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["usage"]["completion_tokens"] == 3

    def test_generate_validation_error_422(self, client):
        # empty prompt
        resp = client.post("/v1/generate", json={"prompt": ""})
        assert resp.status_code == 422

        # invalid temperature
        resp2 = client.post("/v1/generate", json={"prompt": "hi", "temperature": -1.0})
        assert resp2.status_code == 422

        # invalid max_new_tokens
        resp3 = client.post("/v1/generate", json={"prompt": "hi", "max_new_tokens": 0})
        assert resp3.status_code == 422

    def test_engine_not_initialised_503(self):
        """When request.app.state.engine is None, route should return 503."""
        cfg = EngineConfig(model_config_preset="tiny")
        app = create_app(cfg)
        app.state.engine = None

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/health")
            assert resp.status_code == 503
            assert resp.json()["detail"] == "Engine not initialised"

    def test_stream_sse_normal_completion(self, client):
        """Full SSE stream consumption completes with [DONE] and reclaims all resources."""
        with client.stream(
            "POST",
            "/v1/generate",
            json={"prompt": "stream sse test", "max_new_tokens": 4, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            done_seen = False
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        done_seen = True
                        break
                    chunks.append(json.loads(data))

            assert done_seen is True
            assert len(chunks) >= 1
            assert chunks[-1].get("finish_reason") is not None

        metrics = client.get("/metrics").json()
        assert metrics["requests_active"] == 0
        assert metrics["kv_cache_utilization"] == 0.0

    def test_stream_sse_client_disconnect(self, client):
        """Client disconnects early; ensure request is cancelled and cache reclaimed."""
        with client.stream(
            "POST",
            "/v1/generate",
            json={"prompt": "disconnect test", "max_new_tokens": 60, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            # Read first chunk then exit context manager (closing connection)
            for line in resp.iter_lines():
                if line.strip().startswith("data:"):
                    break

        import time
        time.sleep(0.05)
        metrics = client.get("/metrics").json()
        assert metrics["requests_active"] == 0
        assert metrics["kv_cache_utilization"] == 0.0

    def test_concurrent_sse_streams(self, client):
        """Sequential and concurrent streams complete and clean up."""
        for i in range(2):
            with client.stream(
                "POST",
                "/v1/generate",
                json={"prompt": f"concurrent {i}", "max_new_tokens": 3, "stream": True},
            ) as resp:
                assert resp.status_code == 200
                for _ in resp.iter_lines():
                    pass

        metrics = client.get("/metrics").json()
        assert metrics["requests_active"] == 0
        assert metrics["kv_cache_utilization"] == 0.0

    def test_cli_parse_args_defaults(self):
        args = parse_args([])
        assert args.model_config == "tiny"
        assert args.weights is None
        assert args.key_format == "native"
        assert args.device == "cpu"
        assert args.dtype == "float32"
        assert args.port == 8000
        assert args.num_kv_blocks == 128
        assert args.max_batch_size == 32

    def test_cli_parse_args_custom(self):
        custom_argv = [
            "--model-config", "small",
            "--weights", "/tmp/model.safetensors",
            "--key-format", "hf_llama",
            "--device", "cuda",
            "--dtype", "float16",
            "--port", "9000",
            "--num-kv-blocks", "256",
            "--max-batch-size", "16",
            "--log-level", "DEBUG",
        ]
        args = parse_args(custom_argv)
        assert args.model_config == "small"
        assert args.weights == "/tmp/model.safetensors"
        assert args.key_format == "hf_llama"
        assert args.device == "cuda"
        assert args.dtype == "float16"
        assert args.port == 9000
        assert args.num_kv_blocks == 256
        assert args.max_batch_size == 16
        assert args.log_level == "DEBUG"
