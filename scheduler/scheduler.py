from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .batch_manager import BatchManager, StepBatch
from .request import FinishReason, RequestStatus, SamplingParams, SequenceRequest
from .request_queue import RequestQueue

logger = logging.getLogger(__name__)

@dataclass
class SchedulerStats:
    """Running statistics collected across scheduler steps"""
    total_steps:         int   = 0
    total_requests:      int   = 0
    finished_requests:   int   = 0
    preemptions:         int   = 0
    total_prefill_tokens: int  = 0
    total_decode_tokens: int   = 0

    # latency tracking
    ttft_samples:     List[float] = field(default_factory=list)
    latency_samples:  List[float] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> Optional[float]:
        if not self.ttft_samples:
            return None
        return 1000 * sum(self.ttft_samples) / len(self.ttft_samples)

    @property
    def avg_latency_ms(self) -> Optional[float]:
        if not self.latency_samples:
            return None
        return 1000 * sum(self.latency_samples) / len(self.latency_samples)


class Scheduler:
    
    
    def __init__(
        self,
        kv_cache,
        block_size:               int   = 16,
        max_batch_size:           int   = 32,
        max_tokens_per_step:      int   = 4096,
        max_prefill_chunk_tokens: int   = 512,
        starvation_threshold_s:   float = 30.0,
        starvation_reorder_every: int   = 100,   # steps
    ) -> None:
        self._kv_cache  = kv_cache
        self._allocator = kv_cache.allocator
 
        self._queue = RequestQueue(
            starvation_threshold_s=starvation_threshold_s,
        )
        self._batch_manager = BatchManager(
            max_batch_size=max_batch_size,
            max_tokens_per_step=max_tokens_per_step,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            block_size=block_size,
        )
 
        # live state
        self._running:    List[SequenceRequest] = []   # PREFILL or RUNNING
        self._requests:   Dict[str, SequenceRequest] = {}   # id → request
 
        self.stats = SchedulerStats()
        self._starvation_reorder_every = starvation_reorder_every


    # public api called by engine / serving layers

    def add_request(
            self,
            request_id: str,
            prompt_token_ids: List[int],
            sampling_params: Optional[SamplingParams] = None,
            priority: int = 0,
    ) -> SequenceRequest:
        """Add a single request (alias for add_requests for test compatibility)."""
        return self.add_requests(request_id, prompt_token_ids, sampling_params, priority)

    def add_requests(
            self,
            request_id: str,
            prompt_token_ids: List[int],
            sampling_params: Optional[SamplingParams] = None,
            priority: int = 0,
    ) -> SequenceRequest:

        if request_id in self._requests:
            raise ValueError(f"Duplicate request_id: {request_id}")

        req = SequenceRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            priority=priority,
        )

        self._requests[request_id] = req
        self._queue.push(req)
        self.stats.total_requests += 1
        logger.debug("Enqueued %r  prompt_len=%d", request_id, req.prompt_len)

        return req

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request regardless of its current state.
        Return True if found, False if unknown.
        """
        req = self._requests.get(request_id)
        if req is None:
            return False
        if req.is_finished:
            return True

        if req.status == RequestStatus.WAITING:
            self._queue.remove(req)
        elif req.status == RequestStatus.SWAPPED:
            pass
        else:
            self._free_request_blocks(req)
            self._running = [r for r in self._running if r is not req]

        req.cancel()
        logger.info("Cancelled %r", request_id)
        return True

    def get_request(self, request_id: str) -> Optional[SequenceRequest]:
        return self._requests.get(request_id)

    # step CHECK + ADMIT + PREFIL alllocation

    def step(self) -> StepBatch:
        """
        Run one scheduling cycle.  Returns a StepBatch describing exactly
        what the engine should process this iteration.
 
        The engine must call on_step_complete() after running the forward
        pass and sampling, passing a dict of {request_id: new_token_id}
        for every sequence in the batch.
        """

        self.stats.total_steps += 1

        # Anti-starvation reorder (cheap, periodic)
        if self.stats.total_steps % self._starvation_reorder_every == 0:
            n = sum(1 for _ in self._queue)  # iteration triggers starvation reorder
            if n:
                logger.debug("Starvation boosts applied to %d requests", n)

        self._check_finished()

        batch, preempted = self._batch_manager.build_step_batch(
            running=self._running,
            queue=self._queue,
            allocator=self._allocator,
        )

        for req in preempted:
            self._do_preempt(req)

        for req in batch.prefill_requests:
            self._do_admit(req)

        # Update stats
        self.stats.total_prefill_tokens += sum(batch.prefill_token_count)
        self.stats.total_decode_tokens += len(batch.decode_requests)

        logger.debug(
            "step=%d  %r  running=%d  queue=%d  free_blocks=%d",
            self.stats.total_steps,
            batch,
            len(self._running),
            len(self._queue),
            self._allocator.num_free_blocks,
        )

        return batch

    def on_step_complete(self, new_tokens: Dict[str, int]) -> List[SequenceRequest]:

        just_finished: List[SequenceRequest] = []

        for req_id, token_id in new_tokens.items():
            req = self._requests.get(req_id)
            if req is None or req.is_finished:
                continue

            if req.status == RequestStatus.PREFILL:
                req.start_running()
                if req.ttft is not None:
                    self.stats.ttft_samples.append(req.ttft)

            req.append_token(token_id)
            if req.block_table is not None:
                req.block_table.allocate_next_token()

            # stop check
            reason = req.should_stop()
            if reason is not None:
                self._finish_request(req, reason)
                just_finished.append(req)

        return just_finished


    # internal helpers
    def _check_finished(self) -> None:
        """Remove any requests that somehow reached FINISHED from running"""

        still_running = []
        for req in self._running:
            if req.is_finished:
                pass
            else:
                still_running.append(req)
        self._running = still_running

    def _do_admit(self, req: SequenceRequest) -> None:
        """
        Transition WAITING → PREFILL:
          - call req.admit()
          - create a BlockTable
          - allocate KV blocks for the prompt
        """

        req.admit()
        # Create per-sequence block table
        from memory.page_table import BlockTable   # late import avoids circular dep
        bt = BlockTable(seq_id=req.request_id, allocator=self._allocator, block_size=self._batch_manager.block_size)
        bt.allocate_for_prompt(req.prompt_len)
        req.block_table = bt
        self._running.append(req)
        logger.debug("Admitted %r  blocks=%d", req.request_id, self._allocator.num_used_blocks)

    def _do_preempt(self, req: SequenceRequest) -> None:
        """
        Transition RUNNING → SWAPPED:
          - free KV blocks
          - call req.preempt()
          - re-queue as WAITING
        """
        self._free_request_blocks(req)
        req.preempt()
        req.requeue()
        self._queue.push(req)
        self._running = [r for r in self._running if r is not req]
        self.stats.preemptions += 1
        logger.info("Preempted %r (swap_count=%d)", req.request_id, req.swap_count)


    def _finish_request(self, req: SequenceRequest, reason: FinishReason) -> None:
        """Transition to FINISHED and free resources."""
        self._free_request_blocks(req)
        req.finish(reason)
        self._running = [r for r in self._running if r is not req]
        self.stats.finished_requests += 1
        if req.latency is not None:
            self.stats.latency_samples.append(req.latency)
        logger.debug(
            "Finished %r  reason=%s  generated=%d  latency=%.3fs",
            req.request_id,
            reason.value,
            req.num_generated,
            req.latency or 0.0,
        )
 
    def _free_request_blocks(self, req: SequenceRequest) -> None:
        if req.block_table is not None:
            req.block_table.free_all()
            req.block_table = None

    # Observation
    @property
    def num_running(self) -> int:
        return len(self._running)
 
    @property
    def num_waiting(self) -> int:
        return len(self._queue)
 
    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free_blocks
 
    def is_idle(self) -> bool:
        """True when there are no running or waiting requests."""
        return not self._running and not self._queue
 
    def summary(self) -> str:
        s = self.stats
        return (
            f"Scheduler | steps={s.total_steps} "
            f"running={self.num_running} waiting={self.num_waiting} "
            f"finished={s.finished_requests}/{s.total_requests} "
            f"preemptions={s.preemptions} "
            f"free_blocks={self.num_free_blocks} "
            f"avg_ttft={s.avg_ttft_ms:.1f}ms"
            if s.avg_ttft_ms is not None
            else f"Scheduler | steps={s.total_steps} no completions yet"
        )

 





            
            
