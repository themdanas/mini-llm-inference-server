"""
scheduler/batch_manager.py

Batch Manager - decides which requests run each step, enforcing
two hard resource budgets
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .request import FinishReason, RequestStatus, SequenceRequest
from .request_queue import RequestQueue


@dataclass
class StepBatch:
    """The work packet the scheduler hands to the engine for one iteration

    prefill_requests: just admitted; run full (chunked) prompt pass
    decode_requests: already in running state; run one step for each
    prefill_token_count: how many tokens to process for each prefill req
    """

    prefill_requests: List[SequenceRequest] = field(default_factory=list)
    decode_requests: List[SequenceRequest] = field(default_factory=list)
    prefill_token_count: List[int] = field(default_factory=list)

    @property
    def total_sequences(self) -> int:
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        return sum(self.prefill_token_count) + len(self.decode_requests)

    @property
    def is_empty(self) -> bool:
        return self.total_sequences == 0

    def __repr__(self) -> str:
        return (
            f"StepBatch(prefill={len(self.prefill_requests)}, "
            f"decode={len(self.decode_requests)}, "
            f"tokens={self.total_tokens})"
        )


class BatchManager:
    """
    manages the transitions from queue -> running batch each scheduler step
    """

    def __init__(self,
                 max_batch_size: int = 32,
                 max_tokens_per_step: int = 4096,
                 max_prefill_chunk_tokens: int = 512,
                 block_size: int = 16,
                 ) -> None:

        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens
        self.block_size = block_size

    # Main entry point
    def build_step_batch(
            self,
            running: List[SequenceRequest],
            queue: RequestQueue,
            allocator,
    ) -> Tuple[StepBatch, List[SequenceRequest]]:

        batch = StepBatch()
        preempted: List[SequenceRequest] = []

        # decode slots for currently running sequences
        token_budget = self.max_tokens_per_step
        seq_budget = self.max_batch_size

        for req in running:
            if req.status in (RequestStatus.RUNNING, RequestStatus.PREFILL):
                batch.decode_requests.append(req)
                token_budget -= 1
                seq_budget -= 1

        if queue and self._should_preempt(allocator, queue):
            victim = self._pick_preemption_victim(batch.decode_requests)
            if victim is not None:
                batch.decode_requests.remove(victim)
                preempted.append(victim)
                token_budget += 1
                seq_budget += 1

        # admit from queue while budget allow
        while queue and seq_budget > 0 and token_budget > 0:
            next_req = queue.peek()
            if next_req is None:
                break

            chunk_tokens = self._chunk_size(next_req)
            if chunk_tokens > token_budget:
                break

            block_needed = math.ceil(next_req.prompt_len / self.block_size) + 1

            if not allocator.can_allocate(block_needed):
                if batch.decode_requests:
                    victim = self._pick_preemption_victim(batch.decode_requests)

                    if victim is not None:
                        batch.decode_requests.remove(victim)
                        preempted.append(victim)
                        token_budget += 1
                        seq_budget += 1
                        if not allocator.can_allocate(block_needed):
                            break

                else:
                    break

            req = queue.pop()
            batch.prefill_requests.append(req)
            batch.prefill_token_count.append(chunk_tokens)
            token_budget -= chunk_tokens
            seq_budget -= 1

        return batch, preempted

    # helpers

    def _chunk_size(self, req: SequenceRequest) -> int:
        """
        How many prompt tokens to process for `req` this step.
        Accounts for tokens already prefilled in a previous chunked step.
        """
        already_prefilled = getattr(req, "_prefilled_tokens", 0)
        remaining = req.prompt_len - already_prefilled
        return min(remaining, self.max_prefill_chunk_tokens)

    def _should_preempt(self, allocator, queue: RequestQueue) -> bool:
        """
        True if we should consider preempting a running sequence to
        free blocks for the head-of-line waiting request.
        """
        if not queue:
            return False
        next_req = queue.peek()
        if next_req is None:
            return False
        blocks_needed = math.ceil(next_req.prompt_len / self.block_size) + 1
        return not allocator.can_allocate(blocks_needed)

    def _pick_preemption_victim(
        self,
        candidates: List[SequenceRequest],
    ) -> Optional[SequenceRequest]:
        """
        Choose the sequence to preempt using LRU policy:
        the most recently admitted sequence (largest admit_time).
        This is cheapest to recompute and least likely to finish soon.
        """
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: r.admit_time if r.admit_time is not None else float("inf"),
        )

    def blocks_needed(self, req: SequenceRequest) -> int:
        """Blocks required for prompt + 1 decode step."""
        return math.ceil(req.prompt_len / self.block_size) + 1

    def decode_blocks_needed(self, req: SequenceRequest) -> int:
        """Extra blocks needed for next decode step (0 or 1)."""
        total_so_far = req.total_len
        # does the next token spill into a new block?
        if total_so_far % self.block_size == 0:
            return 1
        return 0