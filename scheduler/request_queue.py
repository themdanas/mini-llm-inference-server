from __future__ import annotations

import heapq
import itertools
import time
from typing import Iterator, List, Optional

from .request import SequenceRequest, RequestStatus


class RequestQueue:
    """
    Thread-unsafe priority queue for waiting SequenceRequests.
    All methods must be called from the scheduler's event loop thread.
    """

    def __init__(
            self,
            starvation_threshold_s: float = 30.0,
            starvation_boost: int = 10,
            ) -> None:

        self._heap: List = []
        self._counter = itertools.count()
        self.starvation_threshold_s = starvation_threshold_s
        self.starvation_boost = starvation_boost

    # core operations
    def push(self, request: SequenceRequest) -> None:
        """Add request to the queue"""
        if request.status != RequestStatus.WAITING:
            raise ValueError(
                f"only WAITING requests can be queued: got {request.status}"
            )

        seq_no = next(self._counter)
        eff = self._effective_priority(request)
        heapq.heappush(self._heap, (-eff, request.arrival_time, seq_no, request))

    def pop(self) -> SequenceRequest:
        """Remove and return the next request"""
        if not self._heap:
            raise IndexError("pop from empty RequestQueue")
        return heapq.heappop(self._heap)[3]

    def peek(self) -> Optional[SequenceRequest]:
        """Return the next request without removing it"""
        if not self._heap:
            return None
        return self._heap[0][3]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __iter__(self) -> Iterator[SequenceRequest]:
        now = time.monotonic()
        boosted = 0
        new_heap = []
        for _, arrival, seq_no, req in self._heap:
            eff = self._effective_priority(req, now=now)
            new_heap.append((-eff, arrival, seq_no, req))
            wait = now - req.arrival_time
            if wait > self.starvation_threshold_s:
                boosted += 1
        heapq.heapify(new_heap)
        self._heap = new_heap
        yield from (item[3] for item in self._heap)

    def _effective_priority(
        self,
        request: SequenceRequest,
        now: Optional[float] = None,
    ) -> int:
        if now is None:
            now = time.monotonic()
        wait = now - request.arrival_time
        boost = self.starvation_boost if wait > self.starvation_threshold_s else 0
        return request.priority + boost

    # Inspection
    def max_wait_time(self) -> float:
        """Return maximum waiting time (seconds) among queued requests."""
        if not self._heap:
            return 0.0
        now = time.monotonic()
        return max(now - req.arrival_time for _, _, _, req in self._heap)

    def remove(self, request: SequenceRequest) -> bool:
        for i, (_, _, _, req) in enumerate(self._heap):
            if req is request:
                self._heap[i] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                return True
        return False

    def __repr__(self) -> str:
        return f"RequestQueue(size={len(self)}, max_wait={self.max_wait_time():.1f}s)"