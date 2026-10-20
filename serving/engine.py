"""
serving/engine.py
─────────────────
InferenceEngine — the bridge between the HTTP serving layer and the
inference pipeline (model → scheduler → sampler → tokenizer).

Responsibilities:
  • Load and own the Transformer model, tokenizer, and scheduler
  • Accept generation requests via ``generate()`` / ``generate_stream()``
  • Run an async background step loop that drives the Scheduler
  • Push tokens into per-request ``RequestStream`` queues (for SSE)
  • Expose ``health()`` and ``metrics()`` for observability
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from model.config import ModelConfig
from model.transformer import Transformer
from model.attention import KVCache
from model.weights import load_weights

from tokenizer.tokenizer import BPETokenizer

from sampler.sampler import LogitsProcessor, SamplerConfig

from scheduler.scheduler import Scheduler
from scheduler.request import SamplingParams, FinishReason

from memory.kv_cache import PagedKVCache

from serving.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    StreamChunk,
    UsageInfo,
)
from serving.streaming import RequestStream

logger = logging.getLogger(__name__)


# ─── Engine configuration ──────────────────────────────────────────────────


@dataclass
class EngineConfig:
    """All tunables for the InferenceEngine, gathered in one place."""

    # model
    model_config_preset: str = "tiny"            # "tiny" | "small" | path to json
    model_weights_path: Optional[str] = None     # None → random init
    model_key_format: str = "native"             # "native" | "hf_llama"
    dtype: str = "float32"                       # "float32" | "float16" | "bfloat16"
    device: str = "cpu"                          # "cpu" | "cuda" | "mps"

    # tokenizer
    tokenizer_path: Optional[str] = None         # HF model name or None → tiny

    # scheduler / memory
    block_size: int = 16
    num_kv_blocks: int = 128
    max_batch_size: int = 32
    max_tokens_per_step: int = 4096

    # step loop
    step_interval_s: float = 0.001               # sleep between idle steps


# ─── InferenceEngine ────────────────────────────────────────────────────────


class InferenceEngine:
    """
    Async inference engine.

    Call ``await engine.start()`` once during app lifespan.
    Then submit work via ``generate()`` or ``generate_stream()``.
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._running = False
        self._start_time: float = 0.0
        self._step_task: Optional[asyncio.Task] = None

        # components (initialised in start())
        self._model: Optional[Transformer] = None
        self._tokenizer: Optional[BPETokenizer] = None
        self._scheduler: Optional[Scheduler] = None
        self._kv_cache: Optional[PagedKVCache] = None
        self._model_config: Optional[ModelConfig] = None

        # per-request state managed by the engine
        self._request_kv_caches: Dict[str, List[KVCache]] = {}
        self._request_streams: Dict[str, RequestStream] = {}
        self._request_completions: Dict[str, asyncio.Event] = {}
        self._request_results: Dict[str, GenerateResponse] = {}
        self._request_samplers: Dict[str, LogitsProcessor] = {}
        self._request_prompt_lens: Dict[str, int] = {}
        self._request_errors: Dict[str, str] = {}
        self._streaming_requests: set[str] = set()

        # running token count for metrics
        self._tokens_generated: int = 0

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Load model + tokenizer, create scheduler, start step loop."""
        if self._running:
            return

        logger.info("Engine starting …")
        self._start_time = time.monotonic()

        # 1 — Model
        self._model_config = self._build_model_config()
        self._model = Transformer(self._model_config)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.config.model_weights_path is not None:
            load_weights(
                self._model,
                self.config.model_weights_path,
                key_format=self.config.model_key_format,
                dtype=dtype_map.get(self.config.dtype),
                device=torch.device(self.config.device),
            )

        device = torch.device(self.config.device)
        self._model = self._model.to(device)
        self._model.eval()

        # 2 — Tokenizer
        if self.config.tokenizer_path:
            self._tokenizer = BPETokenizer.from_huggingface(
                self.config.tokenizer_path, add_bos=True, add_eos=False,
            )
        else:
            self._tokenizer = BPETokenizer.tiny_test()

        # 3 — Paged KV cache + Scheduler
        self._kv_cache = PagedKVCache(
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
            n_layers=self._model_config.n_layers,
            n_kv_heads=self._model_config.n_kv_heads,
            d_head=self._model_config.d_head,
            dtype=dtype_map.get(self.config.dtype, torch.float32),
            device=device,
        )

        self._scheduler = Scheduler(
            kv_cache=self._kv_cache,
            block_size=self.config.block_size,
            max_batch_size=self.config.max_batch_size,
            max_tokens_per_step=self.config.max_tokens_per_step,
        )

        # 4 — Background step loop
        self._running = True
        self._step_task = asyncio.create_task(self._step_loop())

        logger.info(
            "Engine ready — model=%s  device=%s  tokenizer=%r",
            self._model_config, self.config.device, self._tokenizer,
        )

    async def shutdown(self) -> None:
        """Stop step loop and clean up."""
        logger.info("Engine shutting down …")
        self._running = False
        if self._step_task is not None:
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass
        logger.info("Engine stopped.")

    # ── Public API ──────────────────────────────────────────────────────

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Non-streaming generation.  Blocks until the request finishes.

        This method submits the request to the scheduler and then polls
        the step loop to advance work.  This avoids reliance on a
        concurrently-running background task, which can deadlock under
        sync test clients (e.g. FastAPI ``TestClient``).
        """
        if not self._running:
            raise RuntimeError("InferenceEngine is not started. Call await engine.start() first.")

        request_id = self._submit_request(request)
        done_event = self._request_completions[request_id]

        try:
            # drive the step loop from here until our request completes
            while not done_event.is_set():
                await self._run_one_step()
                # tiny sleep to let other coroutines run
                await asyncio.sleep(0)

            if request_id in self._request_errors:
                err = self._request_errors.pop(request_id)
                raise RuntimeError(f"Generation failed: {err}")

            result = self._request_results.pop(request_id, None)
            if result is None:
                raise RuntimeError("Request was cancelled or terminated unexpectedly.")
            return result
        finally:
            self._cleanup_request(request_id)

    async def generate_stream(self, request: GenerateRequest) -> RequestStream:
        """Streaming generation.  Returns a ``RequestStream`` immediately."""
        if not self._running:
            raise RuntimeError("InferenceEngine is not started. Call await engine.start() first.")
        request_id = self._submit_request(request)
        return self._request_streams[request_id]

    def cancel(self, request_id: str, error: str | None = None) -> bool:
        """Cancel an in-flight request and clean up its resources."""
        stream = self._request_streams.get(request_id)
        if stream is not None:
            stream.cancelled = True
            stream.close()
        cancelled = self._scheduler.cancel_request(request_id) if self._scheduler else False
        self._streaming_requests.discard(request_id)

        if error:
            self._request_errors[request_id] = error

        done_event = self._request_completions.get(request_id)
        if done_event is not None:
            done_event.set()

        self._cleanup_request(request_id)
        return cancelled

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok" if self._running else "loading",
            model=repr(self._model_config) if self._model_config else "",
            device=self.config.device,
            uptime_s=round(time.monotonic() - self._start_time, 1) if self._running else 0.0,
        )

    def metrics(self) -> MetricsResponse:
        stats = self._scheduler.stats if self._scheduler else None
        return MetricsResponse(
            requests_total=stats.total_requests if stats else 0,
            requests_active=self._scheduler.num_running if self._scheduler else 0,
            requests_waiting=self._scheduler.num_waiting if self._scheduler else 0,
            tokens_generated=self._tokens_generated,
            avg_ttft_ms=stats.avg_ttft_ms if stats else None,
            avg_latency_ms=stats.avg_latency_ms if stats else None,
            kv_cache_utilization=self._kv_cache.utilization if self._kv_cache else 0.0,
        )

    # ── Internal: request submission ────────────────────────────────────

    def _submit_request(self, request: GenerateRequest) -> str:
        """
        Encode the prompt, build sampling params, and enqueue into the
        scheduler.  Returns the request_id.
        """
        request_id = uuid.uuid4().hex[:12]

        # tokenize
        prompt_ids = self._tokenizer.encode(request.prompt)
        if not prompt_ids:
            prompt_ids = [self._tokenizer.vocab.BOS_ID]

        max_seq_len = self._model_config.max_seq_len if self._model_config else 2048
        if len(prompt_ids) >= max_seq_len:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds model context length ({max_seq_len})."
            )

        # Clamp max_new_tokens so prompt + generation never exceeds model context / RoPE limits
        allowed_new_tokens = max(1, max_seq_len - len(prompt_ids))
        effective_max_new_tokens = min(request.max_new_tokens, allowed_new_tokens)

        self._request_prompt_lens[request_id] = len(prompt_ids)

        # build scheduler-level SamplingParams
        eos_ids = list(request.stop_token_ids)
        if self._model_config and self._model_config.eos_token_id not in eos_ids:
            eos_ids.append(self._model_config.eos_token_id)

        sched_params = SamplingParams(
            max_new_tokens=effective_max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=eos_ids,
            stop_strings=list(request.stop),
        )

        # build sampler processor
        sampler_cfg = SamplerConfig(
            greedy=(request.temperature < 1e-6),
            temperature=max(request.temperature, 1e-6),
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=request.min_p,
            rep_penalty=request.repetition_penalty,
            max_new_tokens=effective_max_new_tokens,
            stop_token_ids=eos_ids,
        )
        self._request_samplers[request_id] = LogitsProcessor(sampler_cfg)

        # create per-request KV caches (one per layer)
        device = torch.device(self.config.device)
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float32)
        kv_caches = [
            KVCache(
                n_kv_heads=self._model_config.n_kv_heads,
                d_head=self._model_config.d_head,
                max_seq_len=self._model_config.max_seq_len,
                dtype=dtype,
                device=device,
            )
            for _ in range(self._model_config.n_layers)
        ]
        self._request_kv_caches[request_id] = kv_caches

        # per-request stream + completion event
        self._request_streams[request_id] = RequestStream(
            request_id,
            on_close=self._on_stream_closed,
        )
        self._request_completions[request_id] = asyncio.Event()

        # enqueue
        if request.stream:
            self._streaming_requests.add(request_id)

        self._scheduler.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            sampling_params=sched_params,
            priority=request.priority,
        )

        logger.info(
            "Submitted %s  prompt_len=%d  max_new=%d  stream=%s",
            request_id, len(prompt_ids), request.max_new_tokens, request.stream,
        )
        return request_id

    # ── Internal: single step execution ─────────────────────────────────

    async def _run_one_step(self) -> None:
        """
        Execute a single scheduler step: schedule → forward → sample → update.

        Factored out so it can be called both by the background step loop
        (for streaming / concurrent serving) and inline by ``generate()``
        (to avoid deadlock in sync test clients).
        """
        if self._scheduler.is_idle():
            return

        batch = self._scheduler.step()
        if batch.is_empty:
            return

        new_tokens: Dict[str, int] = {}

        # ── Handle prefill sequences ─────────────────────────────
        for req in batch.prefill_requests:
            rid = req.request_id
            if req.is_finished:
                continue
            kv_caches = self._request_kv_caches.get(rid)
            if kv_caches is None:
                continue

            try:
                token_ids = torch.tensor(
                    [req.prompt_token_ids], dtype=torch.long,
                    device=torch.device(self.config.device),
                )
                with torch.inference_mode():
                    logits = self._model.forward(token_ids, kv_caches=kv_caches)

                # sample from last position
                last_logits = logits[:, -1, :]  # (1, vocab)
                sampler = self._request_samplers.get(rid)
                if sampler is not None:
                    token_id = sampler(last_logits, req.output_token_ids).item()
                else:
                    token_id = last_logits.argmax(dim=-1).item()

                new_tokens[rid] = int(token_id)
            except Exception as exc:
                logger.exception("Failure in prefill for %s: %s", rid, exc)
                self.cancel(rid, error=str(exc))

        # ── Handle decode sequences ──────────────────────────────
        for req in batch.decode_requests:
            rid = req.request_id
            if req.is_finished:
                continue
            kv_caches = self._request_kv_caches.get(rid)
            if kv_caches is None:
                continue

            try:
                # decode: forward only the last generated token
                last_token = req.output_token_ids[-1] if req.output_token_ids else req.prompt_token_ids[-1]
                token_ids = torch.tensor(
                    [[last_token]], dtype=torch.long,
                    device=torch.device(self.config.device),
                )
                with torch.inference_mode():
                    logits = self._model.forward(token_ids, kv_caches=kv_caches)

                last_logits = logits[:, -1, :]
                sampler = self._request_samplers.get(rid)
                if sampler is not None:
                    token_id = sampler(last_logits, req.output_token_ids).item()
                else:
                    token_id = last_logits.argmax(dim=-1).item()

                new_tokens[rid] = int(token_id)
            except Exception as exc:
                logger.exception("Failure in decode for %s: %s", rid, exc)
                self.cancel(rid, error=str(exc))

        # 3 — Update scheduler with generated tokens
        just_finished = self._scheduler.on_step_complete(new_tokens)

        # Check for stop strings
        for rid in new_tokens:
            req = self._scheduler.get_request(rid)
            if req is not None and not req.is_finished and req.sampling_params.stop_strings:
                text_so_far = self._tokenizer.decode(req.output_token_ids)
                matched_stop = False
                for stop_str in req.sampling_params.stop_strings:
                    if stop_str and stop_str in text_so_far:
                        matched_stop = True
                        break
                if matched_stop:
                    self._scheduler._finish_request(req, FinishReason.STOP_STRING)
                    just_finished.append(req)

        # 4 — Push streaming chunks and handle completion
        for rid, token_id in new_tokens.items():
            self._tokens_generated += 1
            stream = self._request_streams.get(rid)
            if stream is None or stream.cancelled:
                if stream and stream.cancelled:
                    self._scheduler.cancel_request(rid)
                    self._streaming_requests.discard(rid)
                    self._cleanup_request(rid)
                continue

            # decode the single token to text
            delta_text = self._tokenizer.decode([token_id])
            chunk = StreamChunk(id=rid, delta=delta_text)
            await stream.push(chunk)

        # 5 — Finalise finished requests
        for req in just_finished:
            await self._finalise_request(req)

    # ── Internal: background step loop ──────────────────────────────────

    async def _step_loop(self) -> None:
        """
        Continuously run scheduler steps.

        The loop primarily serves streaming requests and concurrent work.
        Non-streaming requests driven by ``generate()`` also benefit when
        there are multiple inflight requests.
        """
        logger.info("Step loop started")
        while self._running:
            try:
                if self._scheduler.is_idle():
                    await asyncio.sleep(self.config.step_interval_s)
                    continue

                await self._run_one_step()

                # yield to the event loop
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in step loop")
                await asyncio.sleep(0.1)

        logger.info("Step loop stopped")

    async def _finalise_request(self, req) -> None:
        """Build the final response, signal completion, push stream sentinel."""
        rid = req.request_id
        prompt_len = self._request_prompt_lens.get(rid, 0)

        # decode full output
        full_text = self._tokenizer.decode(req.output_token_ids)

        if req.finish_reason == FinishReason.STOP_STRING and req.sampling_params.stop_strings:
            earliest_idx = None
            for s in req.sampling_params.stop_strings:
                if s:
                    idx = full_text.find(s)
                    if idx != -1:
                        if earliest_idx is None or idx < earliest_idx:
                            earliest_idx = idx
            if earliest_idx is not None:
                full_text = full_text[:earliest_idx]

        # map finish reason
        reason_map = {
            FinishReason.EOS: "eos",
            FinishReason.MAX_TOKENS: "max_tokens",
            FinishReason.STOP_STRING: "stop_string",
            FinishReason.CANCELLED: "cancelled",
        }
        finish_reason = reason_map.get(req.finish_reason, "unknown")

        usage = UsageInfo(
            prompt_tokens=prompt_len,
            completion_tokens=req.num_generated,
            total_tokens=prompt_len + req.num_generated,
        )

        result = GenerateResponse(
            id=rid,
            text=full_text,
            finish_reason=finish_reason,
            usage=usage,
        )

        self._request_results[rid] = result

        # Free generation-only resources immediately
        self._request_kv_caches.pop(rid, None)
        self._request_samplers.pop(rid, None)
        self._request_prompt_lens.pop(rid, None)

        # push final stream chunk with finish_reason, then sentinel
        stream = self._request_streams.get(rid)
        if stream is not None:
            final_chunk = StreamChunk(id=rid, delta="", finish_reason=finish_reason)
            await stream.push(final_chunk)
            await stream.finish()

        # signal the non-streaming waiter
        done_event = self._request_completions.get(rid)
        if done_event is not None:
            done_event.set()

        logger.info(
            "Completed %s  reason=%s  tokens=%d",
            rid, finish_reason, req.num_generated,
        )

    def _on_stream_closed(self, stream: RequestStream) -> None:
        """Called when a stream consumer has finished reading (or disconnected)."""
        rid = stream.request_id
        if stream.cancelled:
            # Client disconnected / cancelled:
            # 1. Cancel request in scheduler to free KV cache blocks immediately
            if self._scheduler is not None:
                self._scheduler.cancel_request(rid)
            # 2. Clean up all per-request state
            self._cleanup_request(rid)
            self._streaming_requests.discard(rid)
        else:
            # Normal completion:
            # Scheduler already freed KV blocks on finish.
            # Generator has now finished sending all chunks to the client.
            # Remove stream and any lingering streaming request state.
            self._request_streams.pop(rid, None)
            self._request_results.pop(rid, None)
            self._request_completions.pop(rid, None)
            self._streaming_requests.discard(rid)

    # ── Internal: cleanup ───────────────────────────────────────────────

    def _cleanup_request(self, request_id: str) -> None:
        """Remove all per-request state."""
        self._request_kv_caches.pop(request_id, None)
        self._request_streams.pop(request_id, None)
        self._request_completions.pop(request_id, None)
        self._request_results.pop(request_id, None)
        self._request_samplers.pop(request_id, None)
        self._request_prompt_lens.pop(request_id, None)

    # ── Internal: model config factory ──────────────────────────────────

    def _build_model_config(self) -> ModelConfig:
        """Resolve the model config from preset name or JSON path."""
        preset = self.config.model_config_preset
        if preset == "tiny":
            return ModelConfig.tiny()
        elif preset == "small":
            return ModelConfig.small()
        else:
            # treat as path to a JSON config file
            return ModelConfig.from_json(preset)
