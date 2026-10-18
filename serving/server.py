"""
serving/server.py
─────────────────
FastAPI application factory, lifespan manager, and CLI entrypoint.

Usage:
    python -m serving.server                          # tiny model, CPU, port 8000
    python -m serving.server --model-config small     # 25M model
    python -m serving.server --model-config ./cfg.json --weights ./model.safetensors
    python -m serving.server --tokenizer meta-llama/Llama-2-7b-hf --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from serving.engine import EngineConfig, InferenceEngine
from serving.routes import router

logger = logging.getLogger(__name__)

# ─── Module-level engine singleton ──────────────────────────────────────────
# Populated by the lifespan handler; accessed by routes via ``app.state``.
_engine: InferenceEngine | None = None


def _create_engine(config: EngineConfig) -> InferenceEngine:
    global _engine
    _engine = InferenceEngine(config)
    return _engine


# ─── Lifespan ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the engine before the first request arrives;
    shut it down when the server stops.
    """
    engine: InferenceEngine | None = getattr(app.state, "engine", None)
    if engine is not None:
        await engine.start()
        logger.info("NanoInfer engine is ready.")
    yield
    if engine is not None:
        await engine.shutdown()
        logger.info("NanoInfer engine shut down cleanly.")


# ─── App factory ────────────────────────────────────────────────────────────


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """
    Build a FastAPI application wired to an ``InferenceEngine``.

    Pass an ``EngineConfig`` to override defaults; otherwise the tiny
    synthetic model is used (useful for tests).
    """
    config = config or EngineConfig()
    engine = _create_engine(config)

    app = FastAPI(
        title="NanoInfer",
        description="Minimal LLM inference server with continuous batching",
        version="0.1.0",
        lifespan=lifespan,
    )

    # stash engine on app state so routes can grab it via Depends
    app.state.engine = engine

    app.include_router(router)

    return app


# ─── CLI entrypoint ─────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nanoinfer",
        description="NanoInfer — Minimal LLM Inference Server",
    )
    p.add_argument(
        "--model-config",
        default="tiny",
        help="Model config preset ('tiny', 'small') or path to a JSON config file",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Path to model weights (.safetensors / .bin / sharded dir)",
    )
    p.add_argument(
        "--key-format",
        default="native",
        choices=["native", "hf_llama"],
        help="Weight key naming convention",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer model name/path (None → tiny synthetic)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for inference",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument(
        "--num-kv-blocks",
        type=int, default=128,
        help="Number of KV cache blocks to pre-allocate",
    )
    p.add_argument(
        "--block-size",
        type=int, default=16,
        help="Tokens per KV cache block",
    )
    p.add_argument(
        "--max-batch-size",
        type=int, default=32,
        help="Maximum number of sequences in a batch",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine_config = EngineConfig(
        model_config_preset=args.model_config,
        model_weights_path=args.weights,
        model_key_format=args.key_format,
        dtype=args.dtype,
        device=args.device,
        tokenizer_path=args.tokenizer,
        num_kv_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
    )

    app = create_app(engine_config)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
