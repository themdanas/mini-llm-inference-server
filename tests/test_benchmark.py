"""
tests/test_benchmark.py
=======================
Test suite for the Phase 5 benchmark module.

Tests cover:
  - RequestRecord derived properties (TTFT, TBT, E2E)
  - percentile computation
  - summarise() aggregation
  - make_random_templates()
  - HarnessConfig defaults
  - StepProfiler context manager
  - Full harness run against a live in-process server (integration)

Run with:
    pytest tests/test_benchmark.py -v
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque

import pytest

from benchmark.metrics import (
    RequestRecord,
    BenchmarkSummary,
    percentiles,
    summarise,
)
from benchmark.harness import (
    HarnessConfig,
    RequestTemplate,
    make_random_templates,
    run as harness_run,
)
from benchmark.profiler import StepProfiler

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from serving.server import create_app

# 1. RequestRecord


class TestRequestRecord:

    def _make_record(self, n_tokens: int = 5) -> RequestRecord:
        t0 = 1000.0
        rec = RequestRecord(
            request_id="r0",
            submit_time=t0,
            prompt_len=10,
        )
        # first token at t0 + 0.050s (50ms TTFT)
        rec.first_token_time = t0 + 0.050
        # tokens every 20ms after that
        for i in range(n_tokens):
            rec.token_times.append(t0 + 0.050 + i * 0.020)
        rec.finish_reason = "max_tokens"
        return rec

    def test_ttft_ms(self):
        rec = self._make_record()
        assert abs(rec.ttft_ms - 50.0) < 0.5

    def test_tbt_ms_list_length(self):
        rec = self._make_record(n_tokens=5)
        # TBT is gap between consecutive tokens → N-1 values
        assert len(rec.tbt_ms_list) == 4

    def test_tbt_values_approx_20ms(self):
        rec = self._make_record(n_tokens=5)
        for tbt in rec.tbt_ms_list:
            assert abs(tbt - 20.0) < 1.0

    def test_e2e_ms(self):
        rec = self._make_record(n_tokens=5)
        # 50ms TTFT + 4 × 20ms TBT = 130ms
        assert abs(rec.e2e_ms - 130.0) < 1.0

    def test_tokens_generated(self):
        rec = self._make_record(n_tokens=7)
        assert rec.tokens_generated == 7

    def test_succeeded_true(self):
        rec = self._make_record()
        assert rec.succeeded is True

    def test_succeeded_false_on_error(self):
        rec = RequestRecord(request_id="r", submit_time=0.0)
        rec.error = "timeout"
        assert rec.succeeded is False

    def test_succeeded_false_on_zero_tokens(self):
        rec = RequestRecord(request_id="r", submit_time=0.0)
        assert rec.succeeded is False

    def test_ttft_none_without_first_token(self):
        rec = RequestRecord(request_id="r", submit_time=0.0)
        assert rec.ttft_ms is None

    def test_tbt_empty_on_single_token(self):
        rec = RequestRecord(request_id="r", submit_time=0.0)
        rec.first_token_time = 0.05
        rec.token_times = [0.05]
        assert rec.tbt_ms_list == []


# 2. Percentile helpers

class TestPercentiles:

    def test_median_of_odd_list(self):
        result = percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
        assert result["p50"] == 30.0

    def test_p99_large_list(self):
        values = list(range(1, 101))   # 1..100
        result = percentiles([float(v) for v in values])
        # 99th percentile of 1–100 should be near 99
        assert 98.0 <= result["p99"] <= 100.0

    def test_min_max(self):
        result = percentiles([5.0, 1.0, 3.0, 2.0, 4.0])
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_empty_list_returns_nan(self):
        result = percentiles([])
        assert math.isnan(result["p50"])

    def test_single_value(self):
        result = percentiles([42.0])
        assert result["p50"] == 42.0
        assert result["p99"] == 42.0

    def test_all_same(self):
        result = percentiles([7.0] * 100)
        assert result["p50"] == 7.0
        assert result["p99"] == 7.0



# 3. Summarise


class TestSummarise:

    def _make_records(self, n: int) -> list:
        records = []
        for i in range(n):
            t0 = float(i)
            rec = RequestRecord(request_id=f"r{i}", submit_time=t0, prompt_len=10)
            rec.first_token_time = t0 + 0.050
            for j in range(8):
                rec.token_times.append(t0 + 0.050 + j * 0.020)
            rec.finish_reason = "max_tokens"
            records.append(rec)
        return records

    def test_succeeded_count(self):
        records = self._make_records(10)
        summary = summarise(records, wall_time_s=2.0, concurrent=4)
        assert summary.succeeded == 10
        assert summary.failed    == 0

    def test_failed_count(self):
        records = self._make_records(8)
        bad = RequestRecord(request_id="bad", submit_time=0.0)
        bad.error = "timeout"
        records.append(bad)
        summary = summarise(records, wall_time_s=2.0, concurrent=4)
        assert summary.failed == 1

    def test_tokens_per_second(self):
        records = self._make_records(10)  # 10 × 8 = 80 tokens
        summary = summarise(records, wall_time_s=1.0, concurrent=4)
        assert abs(summary.tokens_per_second - 80.0) < 0.1

    def test_ttft_in_summary(self):
        records = self._make_records(5)
        summary = summarise(records, wall_time_s=1.0, concurrent=1)
        # TTFT should be around 50ms
        assert 40.0 < summary.ttft_ms["p50"] < 60.0

    def test_avg_output_len(self):
        records = self._make_records(10)  # each has 8 tokens
        summary = summarise(records, wall_time_s=1.0, concurrent=1)
        assert summary.avg_output_len == 8.0

    def test_zero_wall_time_no_crash(self):
        records = self._make_records(5)
        summary = summarise(records, wall_time_s=0.0, concurrent=1)
        assert summary.tokens_per_second == 0.0

    def test_empty_records(self):
        summary = summarise([], wall_time_s=1.0, concurrent=1)
        assert summary.succeeded == 0
        assert summary.total_output_tokens == 0



# 4. Template generation


class TestRequestTemplates:

    def test_make_random_templates_count(self):
        templates = make_random_templates(20)
        assert len(templates) == 20

    def test_prompts_are_non_empty(self):
        templates = make_random_templates(10)
        for t in templates:
            assert len(t.prompt) > 0

    def test_max_new_tokens_in_range(self):
        templates = make_random_templates(50, min_new_tokens=8, max_new_tokens=32)
        for t in templates:
            assert 8 <= t.max_new_tokens <= 32

    def test_stream_default_true(self):
        templates = make_random_templates(5)
        assert all(t.stream for t in templates)



# 5. HarnessConfig defaults


class TestHarnessConfig:

    def test_defaults(self):
        templates = make_random_templates(4)
        cfg = HarnessConfig(templates=templates)
        assert cfg.concurrency  == 8
        assert cfg.arrival_rate == 0.0
        assert cfg.base_url     == "http://localhost:8000"

    def test_custom_concurrency(self):
        templates = make_random_templates(4)
        cfg = HarnessConfig(templates=templates, concurrency=2)
        assert cfg.concurrency == 2


# 6. StepProfiler


class TestStepProfiler:

    def test_records_one_step(self):
        p = StepProfiler()
        with p.step(batch_size=4) as ctx:
            with ctx.phase("scheduling"):
                time.sleep(0.001)
            with ctx.phase("forward"):
                time.sleep(0.002)
        assert len(p.records) == 1

    def test_batch_size_recorded(self):
        p = StepProfiler()
        with p.step(batch_size=7):
            pass
        assert p.records[0].batch_size == 7

    def test_phase_timings_positive(self):
        p = StepProfiler()
        with p.step(batch_size=1) as ctx:
            with ctx.phase("scheduling"):
                time.sleep(0.001)
        rec = p.records[0]
        assert rec.scheduling_ms > 0.0

    def test_total_ms_covers_phases(self):
        p = StepProfiler()
        with p.step(batch_size=1) as ctx:
            with ctx.phase("forward"):
                time.sleep(0.005)
        rec = p.records[0]
        # total_ms should be >= forward_ms
        assert rec.total_ms >= rec.forward_ms

    def test_multiple_steps(self):
        p = StepProfiler()
        for _ in range(5):
            with p.step(batch_size=2):
                pass
        assert len(p.records) == 5

    def test_reset_clears_records(self):
        p = StepProfiler()
        for _ in range(3):
            with p.step(batch_size=1):
                pass
        p.reset()
        assert len(p.records) == 0

    def test_summary_has_all_fields(self):
        p = StepProfiler()
        with p.step(batch_size=4) as ctx:
            with ctx.phase("scheduling"): pass
            with ctx.phase("forward"):    pass
            with ctx.phase("sampling"):   pass
            with ctx.phase("dispatch"):   pass
        s = p.summary()
        assert s.n_steps == 1
        assert "mean" in s.scheduling_ms
        assert "p99"  in s.forward_ms



# 7. Integration: harness against live in-process server


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestHarnessIntegration:
    """
    Fires real HTTP requests against a stub server via ASGITransport.

    This is the end-to-end test: it exercises the full stack —
    harness → HTTP → FastAPI → engine → scheduler → response.
    """

    @pytest.fixture
    def server_url(self, tmp_path):
        """
        Spin up the in-process server and expose it via a real TCP port
        using uvicorn, so the harness (which uses httpx with a real URL)
        can reach it.

        We use a random free port to avoid conflicts between tests.
        """
        import socket
        import uvicorn
        import threading

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        app = create_app()
        config = uvicorn.Config(app, host="127.0.0.1", port=port,
                                log_level="error", lifespan="on")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # wait for server to be ready
        import httpx
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        yield f"http://127.0.0.1:{port}"
        server.should_exit = True
        thread.join(timeout=2.0)

    @pytest.mark.anyio
    async def test_harness_run_completes(self, server_url):
        templates = make_random_templates(4, min_words=2, max_words=4, min_new_tokens=2, max_new_tokens=4)
        config = HarnessConfig(
            templates=templates,
            concurrency=2,
            base_url=server_url,
            timeout_s=30.0,
        )
        summary = await harness_run(config)
        assert summary.total_requests == 4
        assert summary.succeeded >= 1   # at least some must succeed

    @pytest.mark.anyio
    async def test_harness_records_ttft(self, server_url):
        templates = make_random_templates(4, min_words=2, max_words=4, min_new_tokens=2, max_new_tokens=4)
        config = HarnessConfig(
            templates=templates,
            concurrency=2,
            base_url=server_url,
            timeout_s=30.0,
        )
        summary = await harness_run(config)
        # TTFT should be a positive finite number
        assert summary.ttft_ms["p50"] > 0

    @pytest.mark.anyio
    async def test_harness_throughput_positive(self, server_url):
        templates = make_random_templates(4, min_words=2, max_words=4, min_new_tokens=2, max_new_tokens=4)
        config = HarnessConfig(
            templates=templates,
            concurrency=2,
            base_url=server_url,
            timeout_s=30.0,
        )
        summary = await harness_run(config)
        assert summary.tokens_per_second > 0