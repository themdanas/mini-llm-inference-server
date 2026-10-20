"""
benchmark/metrics.py
────────────────────
Statistics aggregation for LLM inference benchmarks.

Core Metrics:
  - TTFT (Time to First Token)
  - TBT (Time Between Tokens)
  - End to End Latency
  - Throughput (tokens/s)
  - Percentiles (P50, P90, P99, min, max)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ─── Per-request recording ───────────────────────────────────────────────────


@dataclass
class RequestRecord:
    request_id: str
    submit_time: float
    prompt_len: int = 0
    first_token_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def tokens_generated(self) -> int:
        return len(self.token_times)

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.submit_time) * 1000.0

    @property
    def tbt_ms_list(self) -> List[float]:
        """Return list of inter-token times in milliseconds."""
        times = self.token_times
        if len(times) < 2:
            return []
        return [
            (times[i + 1] - times[i]) * 1000.0
            for i in range(len(times) - 1)
        ]

    @property
    def tbs_ms_list(self) -> List[float]:
        """Alias for tbt_ms_list."""
        return self.tbt_ms_list

    @property
    def e2e_ms(self) -> Optional[float]:
        """End-to-end latency in milliseconds."""
        if not self.token_times:
            return None
        return (self.token_times[-1] - self.submit_time) * 1000.0

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.tokens_generated > 0


# ─── Percentile Helpers ───────────────────────────────────────────────────────


def _percentile(sorted_values: List[float], p: float) -> float:
    """Linear interpolation between points (C=1)."""
    if not sorted_values:
        return float("nan")
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])

    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= n:
        return float(sorted_values[-1])
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def percentiles(values: List[float]) -> dict:
    """Compute {p50, p90, p99, min, max} for a list of values."""
    if not values:
        return {
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    s = sorted(values)
    return {
        "p50": _percentile(s, 50.0),
        "p90": _percentile(s, 90.0),
        "p99": _percentile(s, 99.0),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


# ─── Summary ─────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkSummary:
    """Aggregated statistics across all requests in a benchmark run."""

    # run metadata
    total_requests: int
    concurrent: int
    wall_time_s: float

    # count
    succeeded: int
    failed: int

    # throughput
    total_output_tokens: int
    tokens_per_second: float

    # TTFT stats (ms)
    ttft_ms: dict  # {p50, p90, p99, min, max}

    # TBT stats (ms)
    tbt_ms: dict  # {p50, p90, p99, min, max}

    # E2E latency stats (ms)
    e2e_ms: dict  # {p50, p90, p99, min, max}

    # Prompt / output length distributions
    avg_prompt_len: float
    avg_output_len: float

    def __str__(self) -> str:
        lines = [
            "=" * 56,
            "  Benchmark Results",
            "=" * 56,
            f"  Requests        : {self.total_requests}  "
            f"({self.succeeded} ok, {self.failed} failed)",
            f"  Concurrency     : {self.concurrent}",
            f"  Wall time       : {self.wall_time_s:.2f}s",
            "",
            f"  Throughput      : {self.tokens_per_second:.1f} tok/s",
            f"  Total tokens    : {self.total_output_tokens}",
            "",
            "  TTFT (ms)       "
            f"  p50={self.ttft_ms['p50']:.1f}  "
            f"p90={self.ttft_ms['p90']:.1f}  "
            f"p99={self.ttft_ms['p99']:.1f}",
            "  TBT  (ms)       "
            f"  p50={self.tbt_ms['p50']:.1f}  "
            f"p90={self.tbt_ms['p90']:.1f}  "
            f"p99={self.tbt_ms['p99']:.1f}",
            "  E2E  (ms)       "
            f"  p50={self.e2e_ms['p50']:.1f}  "
            f"p90={self.e2e_ms['p90']:.1f}  "
            f"p99={self.e2e_ms['p99']:.1f}",
            "",
            f"  Avg prompt len  : {self.avg_prompt_len:.1f} tokens",
            f"  Avg output len  : {self.avg_output_len:.1f} tokens",
            "=" * 56,
        ]
        return "\n".join(lines)


def summarise(
    records: List[RequestRecord],
    wall_time_s: float,
    concurrent: int,
) -> BenchmarkSummary:
    """
    Aggregate a list of RequestRecords into a BenchmarkSummary.

    Parameters
    ----------
    records     : one record per completed (or failed) request
    wall_time_s : total elapsed wall-clock seconds for the whole run
    concurrent  : concurrency level that was used
    """
    good = [r for r in records if r.succeeded]
    bad = [r for r in records if not r.succeeded]

    total_tokens = sum(r.tokens_generated for r in good)
    tps = total_tokens / wall_time_s if wall_time_s > 0 else 0.0

    # Flatten all per-request TBT values into one pool
    all_tbt = [t for r in good for t in r.tbt_ms_list]

    ttft_vals = [r.ttft_ms for r in good if r.ttft_ms is not None]
    e2e_vals = [r.e2e_ms for r in good if r.e2e_ms is not None]

    return BenchmarkSummary(
        total_requests=len(records),
        concurrent=concurrent,
        wall_time_s=wall_time_s,
        succeeded=len(good),
        failed=len(bad),
        total_output_tokens=total_tokens,
        tokens_per_second=tps,
        ttft_ms=percentiles(ttft_vals),
        tbt_ms=percentiles(all_tbt),
        e2e_ms=percentiles(e2e_vals),
        avg_prompt_len=sum(r.prompt_len for r in good) / max(len(good), 1) if good else 0.0,
        avg_output_len=sum(r.tokens_generated for r in good) / max(len(good), 1) if good else 0.0,
    )
