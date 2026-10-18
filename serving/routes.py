"""
serving/routes.py
─────────────────
FastAPI ``APIRouter`` with all HTTP endpoints for NanoInfer.

Routes:
    POST /v1/generate     — text generation (streaming or non-streaming)
    GET  /health          — liveness probe
    GET  /metrics         — scheduler/engine statistics
"""

from __future__ import annotations

import logging
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from serving.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)
from serving.streaming import make_sse_response

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Dependency ─────────────────────────────────────────────────────────────

def get_engine(request: Request):
    """Retrieve the ``InferenceEngine`` from the app state."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return engine


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.post(
    "/v1/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate text from a prompt",
    description=(
        "Submit a prompt for text generation.  "
        "Set ``stream: true`` to receive tokens as Server-Sent Events."
    ),
)
@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    include_in_schema=False,
)
async def generate(
    req: GenerateRequest,
    engine=Depends(get_engine),
):
    """
    Main generation endpoint.

    - **Non-streaming** (default): returns a full ``GenerateResponse`` once
      generation is complete.
    - **Streaming** (``stream: true``): returns an SSE event stream of
      ``StreamChunk`` objects, followed by a ``[DONE]`` sentinel.
    """
    try:
        if req.stream:
            stream = await engine.generate_stream(req)
            return make_sse_response(stream)
        else:
            result = await engine.generate(req)
            return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Generation error for prompt=%r", req.prompt[:50])
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health(engine=Depends(get_engine)):
    """Liveness probe — returns 200 when the engine is ready."""
    return engine.health()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Engine metrics",
)
async def metrics(engine=Depends(get_engine)):
    """
    Returns scheduler and engine statistics:
    active/waiting requests, token throughput, TTFT, latency, KV cache usage.
    """
    return engine.metrics()
