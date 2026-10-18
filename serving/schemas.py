"""
serving/schemas.py
──────────────────
Pydantic v2 models for every HTTP request and response in the
NanoInfer serving API.

Designed to be broadly OpenAI-compatible while staying lightweight.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ─── Request ────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Body for ``POST /v1/generate``."""

    prompt: str = Field(..., min_length=1, description="Input text to generate from")

    # sampling knobs
    max_new_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=100.0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0)

    # stop conditions
    stop: List[str] = Field(default_factory=list, description="Stop strings")
    stop_token_ids: List[int] = Field(default_factory=list)

    # streaming
    stream: bool = False

    # priority (higher = served sooner)
    priority: int = 0


# ─── Response ───────────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    """Token counts returned with every generation response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    """Non-streaming response for ``POST /v1/generate``."""

    id: str = Field(..., description="Unique request identifier (uuid4)")
    text: str = Field(..., description="Full generated text (decoded)")
    finish_reason: str = Field(
        ..., description="eos | max_tokens | stop_string | cancelled"
    )
    usage: UsageInfo


# ─── Streaming ──────────────────────────────────────────────────────────────


class StreamChunk(BaseModel):
    """One SSE event payload during streaming generation."""

    id: str
    delta: str = Field("", description="Newly generated text fragment")
    finish_reason: Optional[str] = Field(
        None, description="Set only on the final chunk"
    )


# ─── Observability ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"] = "ok"
    model: str = ""
    device: str = "cpu"
    uptime_s: float = 0.0


class MetricsResponse(BaseModel):
    requests_total: int = 0
    requests_active: int = 0
    requests_waiting: int = 0
    tokens_generated: int = 0
    avg_ttft_ms: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    kv_cache_utilization: float = 0.0


# ─── Error ──────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx/5xx."""

    error: str
    detail: Optional[str] = None
