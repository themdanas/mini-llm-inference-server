from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class RequestStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    RUNNING = auto()
    SWAPPED = auto()
    FINISHED = auto()


class FinishReason(Enum):
    EOS = "eos"
    MAX_TOKENS = "max_tokens"
    STOP_STRING = "stop_string"
    CANCELLED = "cancelled"


@dataclass
class SamplingParams:
    # Per-request sampling configuration
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be in [0, 1]")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


class SequenceRequest:
    """
    One inference request: carries its prompt, output so far, status,
    and everything the scheduler needs to manage it.

    The BlockTable reference is set by scheduler when the request is
    admitted; it stays None while the request is WAITING or SWAPPED.
    """

    def __init__(
            self,
            request_id: str,
            prompt_token_ids: List[int],
            sampling_params: Optional[SamplingParams] = None,
            priority: int = 0,
            arrival_time: Optional[float] = None,
            ) -> None:

        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.priority = priority
        self.arrival_time = arrival_time if arrival_time is not None else time.monotonic()

        # mutable decode states
        self.output_token_ids: List[int] = []
        self.status: RequestStatus = RequestStatus.WAITING
        self.finish_reason: Optional[FinishReason] = None

        # set by scheduler on admit; cleared on finish/swap
        self.block_table = None

        # timing
        self.admit_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

        # swap counter (how many times preempted)
        self.swap_count: int = 0

        # derived properties

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    @property
    def is_running(self) -> bool:
        return self.status in (RequestStatus.PREFILL, RequestStatus.RUNNING)

    @property
    def ttft(self) -> Optional[float]:
        """Time-to-first-token (seconds)"""
        if self.first_token_time is None or self.admit_time is None:
            return None
        return self.first_token_time - self.admit_time

    @property
    def latency(self) -> Optional[float]:
        """End-to-end latency (seconds). None if not finished."""
        if self.finish_time is None or self.admit_time is None:
            return None
        return self.finish_time - self.admit_time

    # State Transitions
    def admit(self) -> None:
        """WAITING → PREFILL: scheduler is about to run the prompt pass."""
        if self.status != RequestStatus.WAITING:
            raise RuntimeError(
                f"Request {self.request_id}: expected WAITING, got {self.status}"
            )
        self.status = RequestStatus.PREFILL
        self.admit_time = time.monotonic()

    def start_running(self) -> None:
        """PREFILL → RUNNING: prompt pass done, first token emitted."""
        if self.status != RequestStatus.PREFILL:
            raise RuntimeError(
                f"Request {self.request_id}: expected PREFILL, got {self.status}"
            )
        self.status = RequestStatus.RUNNING
        self.first_token_time = time.monotonic()

    def preempt(self) -> None:
        """RUNNING → SWAPPED: evict from batch, free KV blocks."""
        if self.status != RequestStatus.RUNNING:
            raise RuntimeError(
                f"Request {self.request_id}: expected RUNNING, got {self.status}"
            )
        self.status = RequestStatus.SWAPPED
        self.swap_count += 1
        self.block_table = None

    def requeue(self) -> None:
        """SWAPPED → WAITING: put back in queue for re-admission."""
        if self.status != RequestStatus.SWAPPED:
            raise RuntimeError(
                f"Request {self.request_id}: expected SWAPPED, got {self.status}"
            )
        self.status = RequestStatus.WAITING

    def finish(self, reason: FinishReason) -> None:
        """Any → FINISHED."""
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finish_time = time.monotonic()
        self.block_table = None

    def cancel(self) -> None:
        """Immediate cancel from any state."""
        self.finish(FinishReason.CANCELLED)

    # Token append + stop checks
    def append_token(self, token_id: int) -> None:
        """
        Record a newly generated token.
        Does not update status -- caller must call should_stop()"""
        self.output_token_ids.append(token_id)

    def should_stop(self) -> Optional[FinishReason]:
        """
        Check whether the sequence is complete after the last appended token.
        Returns a FinishReason if it should stop, None to keep going.
        """
        sp = self.sampling_params

        # max tokens hard limit
        if self.num_generated >= sp.max_new_tokens:
            return FinishReason.MAX_TOKENS

        # EOS token
        if self.output_token_ids and self.output_token_ids[-1] in sp.stop_token_ids:
            return FinishReason.EOS

        # stop strings (checked externally after detokenisation — placeholder here)
        # This is intentionally left for the serving layer to call after decoding.

        return None

    # repr
    def __repr__(self) -> str:
        return (
            f"SequenceRequest(id={self.request_id!r}, "
            f"status={self.status.name}, "
            f"prompt_len={self.prompt_len}, "
            f"generated={self.num_generated}/{self.sampling_params.max_new_tokens})"
        )

    def __lt__(self, other: "SequenceRequest") -> bool:
        """For priority queue ordering: higher priority first, then FIFO."""
        if self.priority != other.priority:
            return self.priority > other.priority   # higher priority = smaller in heap
        return self.arrival_time < other.arrival_time