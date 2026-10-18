"""
serving/
────────
Public API for the serving package.
"""

from serving.engine import EngineConfig, InferenceEngine
from serving.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    StreamChunk,
    UsageInfo,
    ErrorResponse,
)
from serving.streaming import RequestStream, make_sse_response
from serving.server import create_app

__all__ = [
    # engine
    "EngineConfig",
    "InferenceEngine",
    # schemas
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "MetricsResponse",
    "StreamChunk",
    "UsageInfo",
    "ErrorResponse",
    # streaming
    "RequestStream",
    "make_sse_response",
    # server
    "create_app",
]
