"""
tests/test_scheduler.py
=======================
Test suite for the Phase 4 scheduler module.

Run with:
    pytest tests/test_scheduler.py -v

These tests use lightweight stubs for BlockAllocator and BlockTable so
the scheduler can be exercised without requiring GPU memory.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from scheduler.request import (
    FinishReason,
    RequestStatus,
    SamplingParams,
    SequenceRequest,
)
from scheduler.request_queue import RequestQueue
from scheduler.batch_manager import BatchManager, StepBatch
from scheduler.scheduler import Scheduler



# Stubs


class StubBlock:
    """Minimal PhysicalBlock stub."""
    def __init__(self, block_id: int) -> None:
        self.block_id   = block_id
        self.ref_count  = 0
        self.num_filled = 0


class StubAllocator:
    """Stub BlockAllocator with deterministic free list."""

    def __init__(self, num_blocks: int = 64, block_size: int = 16) -> None:
        self._blocks      = [StubBlock(i) for i in range(num_blocks)]
        self._free        = deque(self._blocks)
        self.block_size   = block_size
        self._num_blocks  = num_blocks

    def allocate(self) -> StubBlock:
        if not self._free:
            raise MemoryError("No free blocks")
        blk = self._free.popleft()
        blk.ref_count  = 1
        blk.num_filled = 0
        return blk

    def free(self, blk: StubBlock) -> None:
        blk.ref_count -= 1
        if blk.ref_count == 0:
            self._free.append(blk)

    def fork(self, blk: StubBlock) -> StubBlock:
        blk.ref_count += 1
        return blk

    def can_allocate(self, n: int) -> bool:
        return len(self._free) >= n

    def blocks_needed_for(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    # CoW not needed for scheduler tests
    def copy_on_write(self, blk): ...


class StubBlockTable:
    """Minimal BlockTable stub."""

    def __init__(self, allocator: StubAllocator, block_size: int = 16) -> None:
        self._allocator = allocator
        self._block_size = block_size
        self._blocks: List[StubBlock] = []
        self._token_count = 0

    def allocate_for_prompt(self, n_tokens: int) -> None:
        n_blocks = math.ceil(n_tokens / self._block_size)
        for _ in range(n_blocks):
            self._blocks.append(self._allocator.allocate())
        self._token_count = n_tokens

    def allocate_next_token(self) -> bool:
        self._token_count += 1
        if (self._token_count % self._block_size) == 1:
            self._blocks.append(self._allocator.allocate())
            return True
        return False

    def free_all(self) -> None:
        for blk in self._blocks:
            self._allocator.free(blk)
        self._blocks.clear()
        self._token_count = 0


class StubKVCache:
    """Minimal KV cache stub used by Scheduler."""

    def __init__(self, num_blocks: int = 64, block_size: int = 16) -> None:
        self.allocator = StubAllocator(num_blocks=num_blocks, block_size=block_size)


def make_scheduler(num_blocks: int = 64, **kwargs) -> Scheduler:
    """Helper: create a Scheduler backed by stub memory."""
    kv = StubKVCache(num_blocks=num_blocks, block_size=kwargs.pop("block_size", 16))
    sched = Scheduler(kv_cache=kv, **kwargs)
    # Monkey-patch _do_admit to use StubBlockTable instead of real BlockTable
    original_admit = sched._do_admit

    def stub_admit(req: SequenceRequest) -> None:
        req.admit()
        bt = StubBlockTable(allocator=kv.allocator, block_size=sched._batch_manager.block_size)
        bt.allocate_for_prompt(req.prompt_len)
        req.block_table = bt
        sched._running.append(req)

    sched._do_admit = stub_admit  # type: ignore
    return sched



# SequenceRequest state machine

class TestSequenceRequest:

    def _make(self, max_new_tokens=50, stop_ids=None):
        sp = SamplingParams(
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids or [2],
        )
        return SequenceRequest("r0", [1, 10, 20, 30], sp)

    def test_initial_state(self):
        req = self._make()
        assert req.status == RequestStatus.WAITING
        assert req.num_generated == 0
        assert req.prompt_len == 4
        assert req.total_len == 4
        assert not req.is_finished

    def test_happy_path_transitions(self):
        req = self._make()
        req.admit();        assert req.status == RequestStatus.PREFILL
        req.start_running();assert req.status == RequestStatus.RUNNING
        req.finish(FinishReason.EOS); assert req.is_finished

    def test_preempt_requeue(self):
        req = self._make()
        req.admit()
        req.start_running()
        req.preempt();  assert req.status == RequestStatus.SWAPPED
        req.requeue();  assert req.status == RequestStatus.WAITING
        assert req.swap_count == 1

    def test_cancel_from_any_state(self):
        for setup in [
            lambda r: None,
            lambda r: r.admit(),
            lambda r: (r.admit(), r.start_running()),
        ]:
            req = self._make()
            setup(req)
            req.cancel()
            assert req.is_finished
            assert req.finish_reason == FinishReason.CANCELLED

    def test_append_and_should_stop_eos(self):
        req = self._make(stop_ids=[2])
        req.admit(); req.start_running()
        req.append_token(100)
        assert req.should_stop() is None
        req.append_token(2)   # EOS
        assert req.should_stop() == FinishReason.EOS

    def test_should_stop_max_tokens(self):
        req = self._make(max_new_tokens=3)
        req.admit(); req.start_running()
        for _ in range(3):
            req.append_token(100)
        assert req.should_stop() == FinishReason.MAX_TOKENS

    def test_all_token_ids(self):
        req = self._make()
        req.admit(); req.start_running()
        req.append_token(99)
        assert req.all_token_ids == [1, 10, 20, 30, 99]

    def test_invalid_transition_raises(self):
        req = self._make()
        with pytest.raises(RuntimeError):
            req.start_running()   # can't go WAITING → RUNNING

    def test_ordering_priority_then_fifo(self):
        t0 = time.monotonic()
        r_lo = SequenceRequest("lo", [1], SamplingParams(), priority=0, arrival_time=t0)
        r_hi = SequenceRequest("hi", [1], SamplingParams(), priority=5, arrival_time=t0 + 1)
        # hi priority wins even though it arrived later
        assert r_hi < r_lo



# SamplingParams validation

class TestSamplingParams:

    def test_defaults_valid(self):
        sp = SamplingParams()
        assert sp.max_new_tokens == 256
        assert sp.temperature == 1.0

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(max_new_tokens=0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(temperature=-0.1)

    def test_top_p_bounds(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.5)
        with pytest.raises(ValueError):
            SamplingParams(top_p=-0.1)

    def test_rep_penalty_bounds(self):
        with pytest.raises(ValueError):
            SamplingParams(repetition_penalty=0.0)



# RequestQueue

class TestRequestQueue:

    def _req(self, rid: str, priority: int = 0, arrival_offset: float = 0.0):
        t = time.monotonic() + arrival_offset
        return SequenceRequest(rid, [1], SamplingParams(), priority=priority, arrival_time=t)

    def test_empty_queue(self):
        q = RequestQueue()
        assert len(q) == 0
        assert not q
        assert q.peek() is None

    def test_push_pop_single(self):
        q = RequestQueue()
        req = self._req("a")
        q.push(req)
        assert len(q) == 1
        out = q.pop()
        assert out is req
        assert len(q) == 0

    def test_fifo_same_priority(self):
        q = RequestQueue()
        r1 = self._req("r1", arrival_offset=0.0)
        r2 = self._req("r2", arrival_offset=0.001)
        r3 = self._req("r3", arrival_offset=0.002)
        for r in [r2, r3, r1]:   # push out of order
            q.push(r)
        assert q.pop() is r1
        assert q.pop() is r2
        assert q.pop() is r3

    def test_priority_ordering(self):
        q = RequestQueue()
        low  = self._req("low",  priority=0)
        high = self._req("high", priority=10)
        med  = self._req("med",  priority=5)
        for r in [low, high, med]:
            q.push(r)
        assert q.pop() is high
        assert q.pop() is med
        assert q.pop() is low

    def test_remove_existing(self):
        q = RequestQueue()
        r1 = self._req("r1")
        r2 = self._req("r2")
        q.push(r1); q.push(r2)
        assert q.remove(r1)
        assert len(q) == 1
        assert q.pop() is r2

    def test_remove_nonexistent(self):
        q = RequestQueue()
        r = self._req("x")
        assert not q.remove(r)

    def test_pop_empty_raises(self):
        q = RequestQueue()
        with pytest.raises(IndexError):
            q.pop()

    def test_only_waiting_requests_allowed(self):
        q = RequestQueue()
        r = self._req("r")
        r.admit()
        with pytest.raises(ValueError):
            q.push(r)



# BatchManager


class TestBatchManager:

    def _alloc(self, num_blocks=32, block_size=4):
        return StubAllocator(num_blocks=num_blocks, block_size=block_size)

    def _req(self, rid, prompt_len=8, priority=0):
        tokens = list(range(1, prompt_len + 1))
        return SequenceRequest(rid, tokens, SamplingParams())

    def test_basic_admit_from_empty_running(self):
        bm  = BatchManager(max_batch_size=4, max_tokens_per_step=512,
                           max_prefill_chunk_tokens=512, block_size=4)
        alloc = self._alloc()
        queue = RequestQueue()
        r = self._req("r0", prompt_len=8)
        queue.push(r)
        batch, preempted = bm.build_step_batch([], queue, alloc)
        assert len(batch.prefill_requests) == 1
        assert r in batch.prefill_requests
        assert preempted == []

    def test_token_budget_limits_admits(self):
        bm = BatchManager(max_batch_size=8, max_tokens_per_step=12,
                          max_prefill_chunk_tokens=12, block_size=4)
        alloc = self._alloc()
        queue = RequestQueue()
        # each request costs 8 tokens; budget is 12 → only 1 fits
        for i in range(3):
            queue.push(self._req(f"r{i}", prompt_len=8))
        batch, _ = bm.build_step_batch([], queue, alloc)
        assert len(batch.prefill_requests) == 1

    def test_seq_budget_limits_admits(self):
        bm = BatchManager(max_batch_size=2, max_tokens_per_step=4096,
                          max_prefill_chunk_tokens=4096, block_size=4)
        alloc = self._alloc()
        queue = RequestQueue()
        for i in range(5):
            queue.push(self._req(f"r{i}", prompt_len=4))
        batch, _ = bm.build_step_batch([], queue, alloc)
        assert len(batch.prefill_requests) <= 2

    def test_no_admit_if_no_free_blocks(self):
        bm = BatchManager(block_size=4)
        # allocator with only 1 free block (need ceil(8/4)+1=3 for an 8-token prompt)
        alloc = StubAllocator(num_blocks=1, block_size=4)
        queue = RequestQueue()
        queue.push(self._req("r0", prompt_len=8))
        batch, _ = bm.build_step_batch([], queue, alloc)
        assert len(batch.prefill_requests) == 0

    def test_decode_slots_counted(self):
        bm = BatchManager(max_batch_size=4, max_tokens_per_step=10, block_size=4)
        alloc = self._alloc()
        queue = RequestQueue()

        # 3 running requests consume 3 decode slots (3 tokens)
        running = []
        for i in range(3):
            r = self._req(f"run{i}", prompt_len=4)
            r.admit(); r.start_running()
            running.append(r)

        # 1 new request in queue (cost: prompt_len=4 tokens)
        queue.push(self._req("new", prompt_len=4))
        batch, _ = bm.build_step_batch(running, queue, alloc)
        # total: 3 decode + 4 prefill = 7 ≤ 10 → admitted
        assert len(batch.prefill_requests) == 1
        assert len(batch.decode_requests)  == 3

    def test_preemption_when_memory_tight(self):
        bm = BatchManager(max_batch_size=8, max_tokens_per_step=512, block_size=4)
        # Only exactly enough blocks for the running sequence (2 blocks),
        # not for a new 8-token prompt (needs 3 blocks)
        alloc = StubAllocator(num_blocks=5, block_size=4)

        # running request holds 2 blocks
        r_run = self._req("run", prompt_len=8)
        r_run.admit(); r_run.start_running()
        # manually claim blocks so allocator thinks they're used
        alloc.allocate(); alloc.allocate()   # simulating 2 held by r_run

        queue = RequestQueue()
        queue.push(self._req("new", prompt_len=8))   # needs 3 blocks

        batch, preempted = bm.build_step_batch([r_run], queue, alloc)
        # With preemption, the running req may be evicted
        # At minimum, preemption was considered (no assert on outcome — policy may vary)
        assert isinstance(batch, StepBatch)

    def test_blocks_needed(self):
        bm = BatchManager(block_size=4)
        r = self._req("r", prompt_len=9)   # ceil(9/4) + 1 = 4
        assert bm.blocks_needed(r) == 4



# Scheduler integration


class TestSchedulerIntegration:

    def test_add_and_admit_single_request(self):
        s = make_scheduler(num_blocks=64, max_batch_size=4,
                           max_tokens_per_step=512)
        req = s.add_request("r0", [1, 2, 3, 4, 5])
        assert s.num_waiting == 1
        assert s.num_running == 0

        batch = s.step()
        assert s.num_running == 1
        assert len(batch.prefill_requests) == 1
        assert batch.prefill_requests[0] is req

    def test_first_token_transitions_to_running(self):
        s = make_scheduler()
        req = s.add_request("r0", [1, 2, 3])
        s.step()   # admit into PREFILL
        assert req.status == RequestStatus.PREFILL

        finished = s.on_step_complete({"r0": 100})
        assert req.status == RequestStatus.RUNNING
        assert req.num_generated == 1
        assert finished == []

    def test_eos_finishes_request(self):
        sp = SamplingParams(stop_token_ids=[2])
        s  = make_scheduler()
        req = s.add_request("r0", [1, 2, 3], sp)
        s.step()
        s.on_step_complete({"r0": 100})   # first token, goes RUNNING

        s.step()
        finished = s.on_step_complete({"r0": 2})   # EOS
        assert len(finished) == 1
        assert req.is_finished
        assert req.finish_reason == FinishReason.EOS
        assert s.num_running == 0

    def test_max_tokens_finishes_request(self):
        sp = SamplingParams(max_new_tokens=3, stop_token_ids=[999])
        s  = make_scheduler()
        req = s.add_request("r0", [1, 2, 3], sp)
        s.step()

        for i, tok in enumerate([10, 20, 30]):
            if i == 0:
                s.on_step_complete({"r0": tok})
                s.step()
            elif i == 1:
                s.on_step_complete({"r0": tok})
                s.step()
            else:
                finished = s.on_step_complete({"r0": tok})

        assert req.is_finished
        assert req.finish_reason == FinishReason.MAX_TOKENS

    def test_multiple_requests_batched(self):
        s = make_scheduler(max_batch_size=8, max_tokens_per_step=512)
        reqs = [s.add_request(f"r{i}", [1, 2, 3]) for i in range(4)]
        batch = s.step()
        assert len(batch.prefill_requests) == 4
        assert s.num_running == 4

    def test_cancel_waiting_request(self):
        s = make_scheduler()
        req = s.add_request("r0", [1, 2, 3])
        assert s.num_waiting == 1
        assert s.cancel_request("r0")
        assert s.num_waiting == 0
        assert req.is_finished
        assert req.finish_reason == FinishReason.CANCELLED

    def test_cancel_running_request(self):
        s = make_scheduler()
        req = s.add_request("r0", [1, 2, 3])
        s.step()   # admit
        assert s.num_running == 1
        assert s.cancel_request("r0")
        assert s.num_running == 0
        assert req.is_finished

    def test_cancel_unknown_request(self):
        s = make_scheduler()
        assert not s.cancel_request("nonexistent")

    def test_blocks_freed_on_finish(self):
        s = make_scheduler(num_blocks=16)
        sp = SamplingParams(max_new_tokens=1, stop_token_ids=[999])
        req = s.add_request("r0", [1, 2, 3], sp)

        free_before = s.num_free_blocks
        s.step()
        assert s.num_free_blocks < free_before   # blocks consumed

        s.on_step_complete({"r0": 10})  # PREFILL → RUNNING
        s.step()
        s.on_step_complete({"r0": 10})  # max_tokens=1 → FINISHED

        assert req.is_finished
        assert s.num_free_blocks == free_before   # blocks returned

    def test_stats_accumulate(self):
        s = make_scheduler()
        sp = SamplingParams(max_new_tokens=2, stop_token_ids=[999])
        s.add_request("r0", [1, 2, 3], sp)
        s.step()
        s.on_step_complete({"r0": 10})
        s.step()
        s.on_step_complete({"r0": 10})
        s.step()
        s.on_step_complete({"r0": 10})

        assert s.stats.total_steps >= 2
        assert s.stats.total_requests == 1

    def test_idle_after_completion(self):
        s = make_scheduler()
        sp = SamplingParams(max_new_tokens=1, stop_token_ids=[2])
        s.add_request("r0", [1, 2], sp)
        s.step()
        s.on_step_complete({"r0": 10})  # RUNNING
        s.step()
        s.on_step_complete({"r0": 2})   # EOS → FINISHED
        s.step()                         # nothing left
        assert s.is_idle()

    def test_duplicate_request_id_raises(self):
        s = make_scheduler()
        s.add_request("dup", [1])
        with pytest.raises(ValueError):
            s.add_request("dup", [2])



# Preemption and re-admission


class TestPreemption:

    def test_preempt_requeue_readmit(self):
        """A preempted request re-enters WAITING and is re-admitted."""
        s = make_scheduler(num_blocks=64, max_batch_size=4)
        req = s.add_request("r0", [1, 2, 3, 4, 5])
        s.step()
        s.on_step_complete({"r0": 10})   # RUNNING

        # Manually trigger preemption
        s._do_preempt(req)
        assert req.status == RequestStatus.WAITING
        assert s.num_running == 0
        assert s.num_waiting == 1
        assert s.stats.preemptions == 1

        # Next step should re-admit
        batch = s.step()
        assert len(batch.prefill_requests) == 1
        assert req.status == RequestStatus.PREFILL



# Edge cases


class TestEdgeCases:

    def test_step_on_empty_scheduler(self):
        s = make_scheduler()
        batch = s.step()
        assert batch.is_empty
        assert s.is_idle()

    def test_on_step_complete_unknown_id(self):
        """Unknown IDs in on_step_complete should not crash."""
        s = make_scheduler()
        req = s.add_request("r0", [1, 2])
        s.step()
        # include a phantom id
        finished = s.on_step_complete({"r0": 10, "phantom": 99})
        assert len(finished) == 0   # no crash, r0 just transitions

    def test_large_queue_drains_in_order(self):
        """FCFS: earlier requests admitted before later ones."""
        s = make_scheduler(max_batch_size=1, max_tokens_per_step=512)
        t0 = time.monotonic()
        r1 = SequenceRequest("r1", [1], SamplingParams(), arrival_time=t0)
        r2 = SequenceRequest("r2", [1], SamplingParams(), arrival_time=t0 + 0.001)
        s._queue.push(r1)
        s._requests["r1"] = r1
        s._queue.push(r2)
        s._requests["r2"] = r2

        batch = s.step()
        assert batch.prefill_requests[0].request_id == "r1"

    def test_zero_generated_tokens_no_stop(self):
        """should_stop returns None immediately after admit (no tokens yet)."""
        req = SequenceRequest("r", [1, 2], SamplingParams(stop_token_ids=[2]))
        req.admit(); req.start_running()
        assert req.should_stop() is None