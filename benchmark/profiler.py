"""
benchmark/profiler.py
Lightwieght profiler wrappper for inference engine

Two kinds of profiling
1. Wall-Clock step profilling (this file no dependencies)
Record how long each scheduler step takes: time spent in scheduling, model forward
pass, sampling and token dispached
Works everyhwhere no GPU Required 

2. Torch profiler (optinal, GPU only)
Records per CUDA-kernal timings, memory allocation events and the
full execution timeline. Produces a trace JSON you can open in chrome://tracing or TensorBoard 
only useful when running a real model on a real GPU
"""

from __future__ import annotations

import time
import statistics
import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


#step timer 
@dataclass
class StepProfile:
    """Timing breakdown for one schedular step"""
    step: int
    batch_size: int
    scheduling_ms: float
    forward_ms: float
    sampling_ms: float
    dispatch_ms: float

    total_ms: float

class StepProfiler:
    
    def __init__(self, max_records: int = 10_000):
        self._records:  List[StepProfile] = []
        self._max      = max_records
        self._step_n   = 0
 
    @contextmanager
    def step(self, batch_size: int):
        """Context manager that wraps one full scheduler step."""
        ctx = _StepContext(self._step_n, batch_size)
        t0 = time.perf_counter()
        try:
            yield ctx
        finally:
            total_ms = (time.perf_counter() - t0) * 1000.0
            profile = StepProfile(
                step=self._step_n,
                batch_size=batch_size,
                scheduling_ms=ctx._phase_ms.get("scheduling", 0.0),
                forward_ms=ctx._phase_ms.get("forward",     0.0),
                sampling_ms=ctx._phase_ms.get("sampling",   0.0),
                dispatch_ms=ctx._phase_ms.get("dispatch",   0.0),
                total_ms=total_ms,
            )
            if len(self._records) < self._max:
                self._records.append(profile)
            self._step_n += 1
 
    @property
    def records(self) -> List[StepProfile]:
        return list(self._records)
 
    def summary(self, last_n: Optional[int] = None) -> "ProfileSummary":
        """Aggregate the last N step profiles (or all if None)."""
        recs = self._records[-last_n:] if last_n else self._records
        return _aggregate(recs)
 
    def reset(self) -> None:
        self._records.clear()
        self._step_n = 0
 
 
class _StepContext:
    """Internal context passed to the `with profiler.step() as ctx` block."""
 
    def __init__(self, step_n: int, batch_size: int):
        self._step_n    = step_n
        self._batch_size = batch_size
        self._phase_ms: Dict[str, float] = {}
 
    @contextmanager
    def phase(self, name: str):
        """Time a named sub-phase of one step."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phase_ms[name] = (time.perf_counter() - t0) * 1000.0
 

 #Aggregate Summary
@dataclass
class ProfileSummary:
    """Aggregated step-level profiling statstics"""
    n_steps: int
    avg_batch_size: float

    scheduling_ms: Dict[str, float]
    forward_ms: Dict[str, float]
    sampling_ms: Dict[str, float]
    dispatch_ms:    Dict[str, float]
    total_ms:       Dict[str, float]
 
    def __str__(self) -> str:
        def fmt(d: dict) -> str:
            return (
                f"mean={d['mean']:.2f}  "
                f"stdev={d['stdev']:.2f}  "
                f"p99={d['p99']:.2f}"
            )
 
        lines = [
            "=" * 60,
            "  Step Profiler Summary",
            "=" * 60,
            f"  Steps profiled : {self.n_steps}",
            f"  Avg batch size : {self.avg_batch_size:.1f}",
            "",
            f"  Scheduling (ms): {fmt(self.scheduling_ms)}",
            f"  Forward    (ms): {fmt(self.forward_ms)}",
            f"  Sampling   (ms): {fmt(self.sampling_ms)}",
            f"  Dispatch   (ms): {fmt(self.dispatch_ms)}",
            f"  Total      (ms): {fmt(self.total_ms)}",
            "=" * 60,
        ]
        return "\n".join(lines)
 
 
def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)
    mean  = sum(s) / n
    stdev = statistics.stdev(s) if n > 1 else 0.0
    idx   = min(int(0.99 * n), n - 1)
    p99   = s[idx]
    return {"mean": mean, "stdev": stdev, "p99": p99}
 
 
def _aggregate(records: List[StepProfile]) -> ProfileSummary:
    if not records:
        return ProfileSummary(
            n_steps=0, avg_batch_size=0.0,
            scheduling_ms=_stats([]), forward_ms=_stats([]),
            sampling_ms=_stats([]),   dispatch_ms=_stats([]),
            total_ms=_stats([]),
        )
    return ProfileSummary(
        n_steps=len(records),
        avg_batch_size=sum(r.batch_size for r in records) / len(records),
        scheduling_ms=_stats([r.scheduling_ms for r in records]),
        forward_ms=   _stats([r.forward_ms    for r in records]),
        sampling_ms=  _stats([r.sampling_ms   for r in records]),
        dispatch_ms=  _stats([r.dispatch_ms   for r in records]),
        total_ms=     _stats([r.total_ms      for r in records]),
    )
 
 

# torch.profiler wrapper (optional — requires torch)

 
class TorchProfilerWrapper:
    """
    Thin wrapper around torch.profiler.profile.
 
    What torch.profiler records
    ---------------------------
    - Every CUDA kernel launch: name, duration, stream, device
    - CPU-side overhead: Python dispatch, ATen ops, memory alloc
    - Memory events: cudaMalloc, cudaFree, peak allocated
    - NCCL events (multi-GPU): all-reduce, broadcast timings
 

    """
 
    def __init__(self, output_dir: str = "/tmp/llm_profile"):
        self._output_dir = output_dir
        self._available  = False
        try:
            import torch
            self._torch     = torch
            self._available = True
        except ImportError:
            logger.warning(
                "torch not available — TorchProfilerWrapper is a no-op. "
                "Install PyTorch to enable GPU kernel profiling."
            )
 
    @contextmanager
    def profile(
        self,
        wait:   int = 1,
        warmup: int = 2,
        active: int = 5,
    ):
        """
        Context manager that wraps N = wait + warmup + active steps.
 
        wait   : steps to skip entirely (no profiling overhead)
        warmup : steps to trace but discard (JIT warm-up)
        active : steps to trace and exporle.
        """
        if not self._available:
            yield
            return
 
        import torch.profiler as tp
        import os
        os.makedirs(self._output_dir, exist_ok=True)
 
        schedule = tp.schedule(wait=wait, warmup=warmup, active=active)
        handler  = tp.tensorboard_trace_handler(self._output_dir)
 
        with tp.profile(
            activities=[
                tp.ProfilerActivity.CPU,
                tp.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,   # stack traces cost ~30% overhead
        ) as prof:
            yield prof
 
        logger.info("Torch profiler trace written to %s", self._output_dir)
 



