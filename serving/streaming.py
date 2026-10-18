"""
serving/streaming.py
────────────────────
Per-request async token queue and Server-Sent Events (SSE) helpers.

The InferenceEngine pushes ``StreamChunk`` objects into a ``RequestStream``.
The FastAPI route wraps the stream as an ``EventSourceResponse`` for the client.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from sse_starlette.sse import EventSourceResponse

from serving.schemas import StreamChunk

logger = logging.getLogger(__name__)


class RequestStream:
    """
    An async iterable queue bridging the engine's step loop to a single
    client's SSE connection.

    Usage
    -----
    1. Engine calls ``await stream.push(chunk)`` for each new token.
    2. Engine calls ``await stream.finish()`` to signal end-of-stream.
    3. The SSE route iterates ``async for chunk in stream: ...``

    Cancellation: the SSE route sets ``stream.cancelled = True`` when the
    client disconnects, and the engine checks this flag each step.
    """

    def __init__(
        self,
        request_id: str,
        on_close: Optional[Callable[[RequestStream], Any]] = None,
    ) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue()
        self.cancelled = False
        self._on_close = on_close
        self._closed = False

    async def push(self, chunk: StreamChunk) -> None:
        """Enqueue a chunk.  Non-blocking in practice (unbounded queue)."""
        if not self.cancelled:
            await self._queue.put(chunk)

    async def finish(self) -> None:
        """Push a ``None`` sentinel to end the stream."""
        await self._queue.put(None)

    def close(self) -> None:
        """Signal that the consumer is done consuming the stream."""
        if not self._closed:
            self._closed = True
            if self._on_close is not None:
                try:
                    res = self._on_close(self)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                except Exception as exc:
                    logger.warning("Error in on_close for %s: %s", self.request_id, exc)

    async def __aiter__(self):
        """Yield chunks until the sentinel ``None`` arrives."""
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            self.close()


def make_sse_response(stream: RequestStream) -> EventSourceResponse:
    """
    Wrap a ``RequestStream`` into an ``sse-starlette`` ``EventSourceResponse``.

    The SSE protocol sends each chunk as a ``data:`` event.  The final
    event carries ``data: [DONE]`` to signal completion (matching OpenAI's
    streaming convention).
    """

    async def _event_generator():
        try:
            async for chunk in stream:
                yield {"data": chunk.model_dump_json()}
            # final sentinel
            yield {"data": "[DONE]"}
        except asyncio.CancelledError:
            stream.cancelled = True
            logger.debug("SSE stream cancelled for %s", stream.request_id)
        finally:
            stream.close()

    return EventSourceResponse(
        _event_generator(),
        media_type="text/event-stream",
    )
