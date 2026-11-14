# NanoInfer — Lightweight LLM Inference Server

<p align="center">
  <b>A production-grade, minimal Large Language Model (LLM) inference server with Continuous Batching, Paged KV Caching, and True Batched Decode built from scratch in PyTorch and Python.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg" alt="Python 3.10 | 3.11">
  <img src="https://img.shields.io/badge/pytorch-2.3%2B-ee4c2c.svg" alt="PyTorch 2.3+">
  <img src="https://img.shields.io/badge/tests-300%20passed-brightgreen.svg" alt="Tests 300 passed">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
</p>

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Benchmark: Naive HuggingFace vs. NanoInfer](#benchmark-naive-huggingface-vs-nanoinfer)
- [System Architecture](#system-architecture)
- [Deep Dive: Core Mechanics](#deep-dive-core-mechanics)
  - [1. Continuous Batching & Iteration-Level Scheduling](#1-continuous-batching--iteration-level-scheduling)
  - [2. Paged KV Cache Memory Management](#2-paged-kv-cache-memory-management)
  - [3. Modern Transformer & True Batched Decode](#3-modern-transformer--true-batched-decode)
  - [4. Real HuggingFace Checkpoints & Tokenizer Support](#4-real-huggingface-checkpoints--tokenizer-support)
  - [5. Composable Logits Processor & Sampling](#5-composable-logits-processor--sampling)
  - [6. Async Serving & Real-Time SSE Streaming](#6-async-serving--real-time-sse-streaming)
  - [7. Interactive Web Dashboard & Telemetry](#7-interactive-web-dashboard--telemetry)
- [Project Layout](#project-layout)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Starting the Server](#starting-the-server)
  - [Interactive Web Dashboard](#interactive-web-dashboard)
  - [Querying the Engine](#querying-the-engine)
  - [Running the Benchmark Harness](#running-the-benchmark-harness)
  - [Running the Test Suite](#running-the-test-suite)
- [API Reference](#api-reference)
- [CLI Reference](#cli-reference)
- [Design Principles](#design-principles)
- [License](#license)

---

## Overview

**NanoInfer** implements the core engineering patterns behind state-of-the-art LLM serving systems (such as vLLM, TGI, and TensorRT-LLM) from first principles in clean, readable Python and PyTorch.

Traditional inference architectures suffer from severe GPU memory fragmentation and idle compute caused by static request batching. NanoInfer solves this by unifying:
1. **Continuous (Iteration-Level) Batching:** Dynamically packs requests at every forward step, interleaving prefill and decode phases without padding bubbles.
2. **Paged KV Cache:** Virtualizes attention key-value states into fixed-size physical memory pages, eliminating external fragmentation and enabling memory overcommit.
3. **True Batched Decode ($B, 1$ Forward Pass):** Vectorized decode engine executing multi-sequence batches in a single model forward pass with per-sequence 4D attention masks and RoPE offsets.
4. **HuggingFace Checkpoint Compatibility:** Natively loads real pretrained models (e.g., `HuggingFaceTB/SmolLM2-135M-Instruct`) in `.safetensors` format with automated weight tying and LLaMA key remapping.
5. **Interactive Observability Dashboard:** Built-in dark glassmorphic web dashboard with real-time SSE playground, dynamic physical block grid visualization, continuous batching monitor, and one-click load benchmarking.
6. **High-Performance Async Server:** FastAPI backend with Server-Sent Events (SSE), background step-loop execution, prompt context limits, and client cancellation handling.

---

## Key Features

- **True Batched Decode:** Eliminates sequential decode loops by evaluating all active decode sequences in a unified $(B, 1)$ tensor forward pass.
- **Zero External KV Fragmentation:** Physical block allocator with logical-to-physical page table mapping saves up to 75% memory over static allocation.
- **Dynamic Iteration Scheduling:** Request-level preemption, token budgeting (`max_tokens_per_step`), and continuous prefill/decode pipelining.
- **Pretrained Model Support:** Out-of-the-box loading for Hugging Face LLaMA/SmolLM architectures alongside lightweight synthetic presets for unit testing.
- **Production Sampling Pipeline:** Temperature scaling, Top-$K$, Top-$P$ (Nucleus), Min-$P$, Repetition Penalty, and multi-token stop strings.
- **Real-Time Observability:** Built-in Prometheus-style metrics (`/metrics`), live memory state inspection (`/api/state`), and interactive web GUI (`/`).

---

## Benchmark: Naive HuggingFace vs. NanoInfer

Real-world apples-to-apples benchmark running `HuggingFaceTB/SmolLM2-135M-Instruct` on CPU comparing naive HuggingFace Transformers (`model.generate()` with contiguous KV allocation) against NanoInfer:

| Engine | Concurrency | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | E2E P99 (ms) | KV Memory Waste (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive HuggingFace (`model.generate`)** | 1 | **85.8** | 27.5 | 76.0 | 328.2 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 1 | **81.4** | 50.1 | 59.3 | 319.8 | **25.2%** |
| **Naive HuggingFace (`model.generate`)** | 4 | **84.0** | 588.7 | 600.3 | 853.1 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 4 | **96.9** | 187.5 | 251.4 | 1203.8 | **25.2%** |
| **Naive HuggingFace (`model.generate`)** | 8 | **85.2** | 1288.7 | 1716.8 | 1968.1 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 8 | **137.5** | 342.0 | **342.3** | 1422.5 | **25.2%** |

### Key Takeaways:
- **$5\times$ Faster TTFT at Scale:** At Concurrency 8, NanoInfer achieves a $P_{99}$ Time-to-First-Token of **342.3 ms** compared to **1,716.8 ms** for naive Hugging Face.
- **$1.6\times$ Throughput Scaling:** True batched decode scales throughput from 77.0 tok/s up to **137.5 tok/s** at Concurrency 8, whereas naive execution remains bottlenecked at ~85 tok/s.
- **$4\times$ Memory Savings:** Paged KV caching cuts memory waste from **98.3%** down to **25.2%** by allocating physical blocks on-demand instead of padding to `max_seq_len`.

---

## System Architecture

The following diagram illustrates the lifecycle of incoming generation requests through NanoInfer:

```mermaid
flowchart TD
    Client(["Client / Browser / curl / Benchmark"])
    
    subgraph ServingLayer ["Serving Layer (FastAPI & Web UI)"]
        Routes["API Endpoints<br/><code>/generate</code> & <code>/v1/completions</code>"]
        Dashboard["Web Dashboard<br/>(Visual Memory & Batch Monitor)"]
        Engine["InferenceEngine<br/>(Background Step Loop & Lifespan)"]
        Stream["RequestStream<br/>(Async Queue & SSE Formatter)"]
    end
    
    subgraph SchedulerLayer ["Scheduler Layer"]
        Queue["Priority Request Queue<br/>(Waiting / Running / Swapped)"]
        BatchManager["Batch Manager<br/>(Token Budgeting & Preemption)"]
    end
    
    subgraph MemoryLayer ["Memory Layer (PagedAttention)"]
        BlockAlloc["Block Allocator<br/>(Free List & Ref Counts)"]
        PageTable["Page Table<br/>(Logical Seq Blocks ➔ Physical Blocks)"]
        PhysicalPool[("Physical KV Pool<br/>[num_blocks, block_size, heads, d_head]")]
    end
    
    subgraph ModelLayer ["Model Execution Layer"]
        Transformer["LLaMA Transformer<br/>(RoPE + GQA + SwiGLU + Batched Views)"]
        Sampler["LogitsProcessor Pipeline<br/>(Temp + Top-K + Top-P + Min-P + Rep-Penalty)"]
    end

    Client -->|HTTP / SSE Request| Routes
    Client -.->|Browse UI| Dashboard
    Routes --> Engine
    Engine -->|Submit Request| Queue
    
    Engine -->|Drive Step Loop| BatchManager
    BatchManager -->|Query Free Memory| BlockAlloc
    BlockAlloc --> PageTable
    PageTable -.-> PhysicalPool
    
    BatchManager -->|Prefill & Batched Decode| Transformer
    Transformer -->|Batched Logits (B, 1, Vocab)| Sampler
    Sampler -->|Sampled Token IDs| Engine
    
    Engine -->|Push StreamChunk| Stream
    Stream -->|SSE Event Stream| Client
    Engine -->|Update Request State| BatchManager
```

---

## Deep Dive: Core Mechanics

### 1. Continuous Batching & Iteration-Level Scheduling
In standard static batching, all sequences in a batch must wait for the slowest sequence to finish, causing compute bubbles.

NanoInfer implements **continuous (iteration-level) batching** inside [`scheduler/`](file:///Users/mdanas/Programming/Projects/NanoInfer/mini-llm-inference-server/scheduler/):
* **Dynamic Iteration Step:** Instead of batching for an entire generation run, scheduling decisions occur at every forward token step (`_run_one_step`).
* **Prefill / Decode Interleaving:** Completed prefill sequences seamlessly transition to the decode pool, and newly arrived requests immediately join the batch on the next step.
* **Token Budgeting:** The scheduler enforces a configurable `max_tokens_per_step` budget to prevent prefill phases from stalling active decode streams.
* **Resource-Aware Admission:** A waiting request is only admitted into the running batch if the `BlockAllocator` guarantees sufficient memory blocks for its KV cache.

### 2. Paged KV Cache Memory Management
During autoregressive generation, keys and values grow incrementally. Allocating contiguous memory upfront for `max_seq_len` wastes up to 60–80% of memory due to over-allocation and variable sequence lengths.

NanoInfer’s [`memory/`](file:///Users/mdanas/Programming/Projects/NanoInfer/mini-llm-inference-server/memory/) implements virtual memory inspired by OS page tables:
* **Physical KV Cache:** KV activations are pre-allocated into a contiguous memory block pool shaped as `[num_blocks, block_size, n_kv_heads, d_head]`.
* **Block Allocator (`BlockAllocator`):** Manages a free list of physical block IDs. It supports block allocation, ref-counted sharing (for copy-on-write prompt sharing), and instant reclamation.
* **Page Table (`BlockTable`):** Maps a request's logical sequence offsets (divided into fixed blocks of 16 tokens) to non-contiguous physical blocks in the pool.
* **Zero External Fragmentation:** Memory is allocated in uniform chunk sizes (e.g., 16 tokens). When a sequence completes, its blocks immediately return to the free list.

### 3. Modern Transformer & True Batched Decode
The model core in [`model/`](file:///Users/mdanas/Programming/Projects/NanoInfer/mini-llm-inference-server/model/) reflects modern open-weight LLMs (LLaMA-3, SmolLM, Mistral):
* **RMSNorm:** Root Mean Square Layer Normalization applied prior to attention and feed-forward blocks for improved numerical stability.
* **RoPE (Rotary Position Embeddings):** Applied to Query and Key projections, providing relative position awareness across variable context lengths.
* **GQA (Grouped-Query Attention):** Multiple query heads share a single Key/Value head (e.g. 9 Query heads per 3 KV heads in SmolLM2), dramatically reducing memory bandwidth demand.
* **SwiGLU FFN:** Swish-Gated Linear Unit activations for enhanced representation capacity.
* **True Batched Decode ($B, 1$ Forward Pass):**
  - Gathers each active decode sequence's physical blocks via `PagedKVCache.make_batched_decode_views(...)`.
  - Automatically right-aligns/pads historical KV states into `(B, n_kv_heads, max_s, d_head)`.
  - Constructs a 4D additive attention mask of shape `(B, 1, 1, max_s)` with `-inf` on padding tokens.
  - Broadcasts sequence-specific RoPE positions `(B, 1)` and evaluates the entire batch in a single forward pass, eliminating sequential Python loops.

### 4. Real HuggingFace Checkpoints & Tokenizer Support
NanoInfer supports loading real open-weight models directly from HuggingFace Hub or local disk:
* **HuggingFace Checkpoint Loader (`weights.py`):** Automatically downloads and parses `.safetensors` checkpoints, maps HuggingFace LLaMA key names to NanoInfer weights, and handles tied embeddings (`lm_head.weight ← embedding.weight`).
* **HuggingFace Tokenizer Integration (`tokenizer.py`):** `HFTokenizerWrapper` integrates fast HuggingFace tokenizers, ensuring lossless SSE token delta streaming, correct BOS/EOS special token handling, and multi-token vocabulary operations.

### 5. Composable Logits Processor & Sampling
Token generation logic in [`sampler/`](file:///Users/mdanas/Programming/Projects/NanoInfer/mini-llm-inference-server/sampler/) uses a modular pipeline architecture:
* **LogitsProcessor Pipeline:** Logits pass sequentially through configurable processors:
  1. *Repetition Penalty:* Penalizes previously generated tokens.
  2. *Temperature Scaling:* Modulates distribution sharpness (temperature `0.0` triggers greedy argmax).
  3. *Top-K Filtering:* Truncates candidate pool to top $K$ probabilities.
  4. *Top-P (Nucleus) Filtering:* Dynamically selects the minimal set of tokens exceeding cumulative probability $P$.
  5. *Min-P Filtering:* Prunes tokens whose probability is less than a fraction of the maximum token probability.
* **Stop Criteria Engine:** Evaluates end-of-sequence (EOS) token IDs, custom token lists, and dynamic string patterns (stop strings) directly on decoded text.

### 6. Async Serving & Real-Time SSE Streaming
The HTTP layer in [`serving/`](file:///Users/mdanas/Programming/Projects/NanoInfer/mini-llm-inference-server/serving/) provides an enterprise-ready interface:
* **Background Step Loop:** An asynchronous event loop continuously advances the scheduler, decodes tokens, and dispatches them without blocking HTTP threadpools.
* **Server-Sent Events (SSE):** Token deltas stream immediately to clients as `data: {"id": "...", "delta": "...", "finish_reason": null}` chunks.
* **Client Disconnect Reclamation:** When an HTTP client aborts (e.g., closing a tab or sending `SIGINT`), the engine detects the closed socket, aborts the request, and instantly frees its allocated KV blocks.
* **Context Boundary Protection:** Automatically clamps requested generation lengths against model architecture limits to guarantee out-of-bounds safety.

### 7. Interactive Web Dashboard & Telemetry
NanoInfer includes a dark glassmorphic web dashboard served directly at `http://localhost:8000/`:
* **Inference Playground:** Interactive prompt tester with real-time SSE token streaming, temperature, top-p, top-k, and repetition penalty sliders.
* **Live Paged KV Cache Visualizer:** Renders the physical memory block grid in real-time, displaying free blocks, allocated blocks, and active write slots.
* **Continuous Batching Monitor:** Inspects in-flight requests, execution phases (`PREFILL` vs `DECODE`), token counts, and memory usage.
* **One-Click Benchmark Runner:** Fires 8 concurrent load requests directly from the UI and graphs latency and throughput.

---

## Project Layout

```
mini-llm-inference-server/
├── model/                  # Core Transformer Neural Network
│   ├── config.py           # ModelConfig dataclass & presets ('tiny', 'small', 'smollm2_135m')
│   ├── transformer.py      # Transformer & TransformerBlock definitions with attn_mask support
│   ├── attention.py        # MultiHeadAttention, RoPE, GQA, FlashAttention
│   ├── layers.py           # RMSNorm, Embedding, SwiGLU / GELU FFN
│   └── weights.py          # Safetensors loader, PyTorch checkpoints, HF key mapping
│
├── tokenizer/              # Tokenization Pipeline
│   ├── vocabulary.py       # Vocabulary dictionary, merge rules, BPE byte tables
│   └── tokenizer.py        # Byte-level BPE Tokenizer & HFTokenizerWrapper
│
├── sampler/                # Sampling & Logits Processing
│   ├── strategies.py       # Temperature, Top-K, Top-P, Min-P, Rep-Penalty
│   └── sampler.py          # SamplerConfig presets & LogitsProcessor pipeline
│
├── memory/                 # Paged KV Cache Memory Manager
│   ├── block.py            # Physical KV block representation
│   ├── block_allocator.py  # Free-list block allocator with ref-counting
│   ├── page_table.py       # Logical-to-physical sequence page table (BlockTable)
│   └── kv_cache.py         # PagedKVCache, SingleSeqPagedView & BatchedDecodePagedView
│
├── scheduler/              # Continuous Batching Engine
│   ├── request.py          # InferenceRequest lifecycle and sequence state
│   ├── request_queue.py    # Priority-based waiting and running queues
│   ├── batch_manager.py    # Dynamic batch formation and token allocation
│   └── scheduler.py        # Central iteration-level scheduling loop
│
├── serving/                # FastAPI Asynchronous Web Server
│   ├── schemas.py          # Pydantic models for requests, responses, metrics
│   ├── engine.py           # InferenceEngine coordinating scheduler, model & batched decode
│   ├── routes.py           # FastAPI routes (/generate, /v1/completions, /health, /api/state)
│   ├── streaming.py        # RequestStream and SSE formatting utilities
│   └── server.py           # CLI entrypoint & Uvicorn runner
│
├── web/                    # Built-in Observability Dashboard
│   ├── index.html          # Glassmorphic dashboard layout
│   ├── app.js              # SSE streaming client, block visualizer, benchmark runner
│   └── style.css           # Modern dark UI theme
│
├── benchmark/              # Performance Profiler & Harness
│   ├── metrics.py          # TTFT, TBT, E2E percentiles & summary aggregation
│   ├── harness.py          # Async HTTP client load generator & sweep runner
│   ├── profiler.py         # Step profiler & PyTorch profiler wrapper
│   └── compare_hf_vs_nanoinfer.py # Side-by-side benchmark comparison script
│
└── tests/                  # 299 Comprehensive Unit & Integration Tests
    ├── test_model.py       # Transformer, RoPE, Attention, KV equivalence tests
    ├── test_tokenizer.py   # BPE encoding, decoding, round-trip tests
    ├── test_sampler.py     # Deterministic & stochastic sampling assertions
    ├── test_memory.py      # Paged cache allocation, fragmentation, batched view tests
    ├── test_scheduler.py   # Continuous batching, admission, preemption tests
    ├── test_serving.py     # HTTP endpoints, SSE streaming, cancellation, batched decode tests
    ├── test_benchmark.py   # Harness metrics, percentiles, end-to-end load tests
    └── test_e2e.py         # Full-stack integrated generation tests
```

---

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/themdanas/mini-llm-inference-server.git
   cd mini-llm-inference-server
   ```

2. **Set up Python environment:**
   ```bash
   conda create -n infer_env python=3.11 -y
   conda activate infer_env
   pip install -r requirements.txt
   ```

### Starting the Server

#### Option A: Real Pretrained Hugging Face Model (Recommended)
Run NanoInfer with a real HuggingFace checkpoint (automatically downloads weights from HF Hub and allocates 512 KV blocks):
```bash
python -m serving.server --hf-model HuggingFaceTB/SmolLM2-135M-Instruct --port 8000
```

#### Option B: Lightweight Fast Development Model
Run NanoInfer with an in-memory tiny model (CPU, starts in < 1 second):
```bash
python -m serving.server --model-config tiny --port 8000
```

#### Option C: Custom Production Configuration
```bash
python -m serving.server \
  --hf-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --device cpu \
  --dtype float32 \
  --num-kv-blocks 512 \
  --block-size 16 \
  --max-batch-size 32 \
  --port 8000
```

---

### Interactive Web Dashboard

Open **[http://localhost:8000/](http://localhost:8000/)** in your web browser:
- **Inference Playground:** Live token generation with real-time SSE streaming.
- **Physical Memory Grid:** Interactive visualization of physical KV blocks updating in real time.
- **Continuous Batching Telemetry:** Active sequence inspector displaying prefill and decode stages.
- **One-Click Benchmark:** Trigger 8 concurrent benchmark requests directly from your browser.

---

### Querying the Engine

#### 1. Standard Generation via `curl`
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The primary bottleneck in autoregressive LLM inference is",
    "max_new_tokens": 30,
    "temperature": 0.7
  }'
```

#### 2. Real-Time Streaming (SSE)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain continuous batching in one paragraph:",
    "max_new_tokens": 50,
    "stream": true
  }'
```

#### 3. Python Async Client
```python
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "Key-value caching reduces computational redundancy because",
                "max_new_tokens": 30,
                "temperature": 0.0  # Greedy decode
            },
            timeout=30.0,
        )
        data = response.json()
        print(f"Generated text: {data['text']}")
        print(f"Tokens: {data['usage']['completion_tokens']}")

asyncio.run(main())
```

---

### Running the Benchmark Harness

#### 1. Concurrency Load Test
Evaluate generation throughput and latency percentiles under concurrent load:
```bash
python -m benchmark.harness --url http://localhost:8000 --requests 20 --concurrency 4 --max-tokens 32
```

**Example output:**
```
========================================================
  Benchmark Results
========================================================
  Requests        : 20  (20 ok, 0 failed)
  Concurrency     : 4
  Wall time       : 0.14s

  Throughput      : 2210.4 tok/s
  Total tokens    : 310

  TTFT (ms)         p50=8.4   p90=10.2  p99=12.1
  TBT  (ms)         p50=0.8   p90=1.1   p99=1.5
  E2E  (ms)         p50=24.5  p90=42.1  p99=46.3

  Avg prompt len  : 6.2 tokens
  Avg output len  : 15.5 tokens
========================================================
```

#### 2. Automated Concurrency Sweep
Evaluate scaling across concurrency levels `[1, 2, 4, 8, 16]`:
```bash
python -m benchmark.harness --url http://localhost:8000 --requests 50 --sweep
```

#### 3. Naive HuggingFace vs NanoInfer Comparison
Run an apples-to-apples comparison script across concurrency levels:
```bash
# Terminal 1: Start NanoInfer with SmolLM2
python -m serving.server --hf-model HuggingFaceTB/SmolLM2-135M-Instruct --port 8000

# Terminal 2: Run comparison sweep
python -m benchmark.compare_hf_vs_nanoinfer --hf-model HuggingFaceTB/SmolLM2-135M-Instruct --url http://localhost:8000
```

---

### Running the Test Suite

NanoInfer includes **299 comprehensive unit, component, and integration tests** covering attention mechanics, tokenizer round-trips, block allocators, continuous scheduling, batched decode equivalence, and FastAPI HTTP/SSE endpoints:

```bash
# Run all tests
pytest tests/ -v

# Run memory and batched decode tests
pytest tests/test_memory.py -v

# Run serving and continuous batching tests
pytest tests/test_serving.py -v

# Run end-to-end integration tests
pytest tests/test_e2e.py -v
```

---

## API Reference

### `POST /generate` & `POST /v1/completions`

#### Request Parameters
| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | **Required** | The input prompt text. |
| `max_new_tokens` | `integer` | `64` | Maximum number of new tokens to generate. |
| `temperature` | `float` | `1.0` | Sampling temperature (`0.0` triggers greedy decoding). |
| `top_p` | `float` | `1.0` | Nucleus sampling cumulative probability threshold. |
| `top_k` | `integer` | `0` | Top-$K$ filtering threshold (`0` disables). |
| `min_p` | `float` | `0.0` | Min-$P$ threshold relative to max token probability. |
| `repetition_penalty` | `float` | `1.0` | Penalty applied to repeated tokens (`1.0` disables). |
| `stop` | `list[str]` | `[]` | List of string sequences that halt generation. |
| `stop_token_ids` | `list[int]` | `[]` | List of token IDs that halt generation. |
| `stream` | `boolean` | `false` | Enable Server-Sent Events (SSE) streaming. |

#### Response (Non-Streaming)
```json
{
  "id": "a1b2c3d4e5f6",
  "text": "Generated text completion goes here...",
  "finish_reason": "max_tokens",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 30,
    "total_tokens": 42
  }
}
```

### `GET /health`
Returns system liveness status, active model architecture, and uptime:
```json
{
  "status": "ok",
  "model": "ModelConfig(layers=30, d_model=576, heads=9/3, d_ff=1536, vocab=49152, ~134.5M params)",
  "device": "cpu",
  "uptime_s": 240.5
}
```

### `GET /metrics`
Provides Prometheus-style telemetry:
```json
{
  "requests_total": 142,
  "requests_active": 4,
  "requests_waiting": 0,
  "tokens_generated": 3410,
  "avg_ttft_ms": 9.2,
  "avg_latency_ms": 32.8,
  "kv_cache_utilization": 0.125
}
```

### `GET /api/state`
Returns real-time engine telemetry powering the visual dashboard:
```json
{
  "memory": {
    "total_blocks": 512,
    "free_blocks": 496,
    "used_blocks": 16,
    "utilization": 0.03125,
    "blocks": [
      {"id": 0, "state": "allocated", "ref_count": 1, "num_filled": 16, "request_id": "a1b2c3d4e5f6"}
    ]
  },
  "scheduler": {
    "num_running": 1,
    "num_waiting": 0,
    "steps": 42,
    "preemptions": 0,
    "running": [
      {"id": "a1b2c3d4e5f6", "phase": "RUNNING", "prompt_len": 12, "generated_len": 8, "block_ids": [0]}
    ],
    "waiting": []
  },
  "metrics": {
    "requests_total": 12,
    "tokens_generated": 184,
    "avg_ttft_ms": 8.4,
    "avg_latency_ms": 28.1,
    "uptime_s": 120.0
  }
}
```

### `GET /api/comparison`
Returns empirical benchmark comparison data between Naive HuggingFace (`model.generate`) and NanoInfer on `HuggingFaceTB/SmolLM2-135M-Instruct`:
```json
{
  "model_info": "HuggingFaceTB/SmolLM2-135M-Instruct (30 layers, d_model=576, GQA 9/3 heads, ~135M params)",
  "headline": {
    "ttft_speedup": {
      "value": "5.0x Faster",
      "subtitle": "P99 Time-To-First-Token at 8 concurrent users: 342.3ms vs 1716.8ms"
    },
    "memory_saved": {
      "value": "73.1% Less KV Waste",
      "subtitle": "Paged 16-token blocks (25.2% waste) vs Contiguous 2048-slot reservation (98.3% waste)"
    },
    "throughput_gain": {
      "value": "+38.0% Throughput",
      "subtitle": "117.6 tok/s vs 85.2 tok/s via Batched (B, 1) Decode"
    },
    "e2e_reduction": {
      "value": "-27.7% Tail Latency",
      "subtitle": "P99 End-to-End completion reduced from 1968.1ms to 1422.5ms"
    }
  },
  "rows": [
    {
      "concurrency": 1,
      "naive_hf": {"throughput": 85.8, "ttft_p50": 27.5, "ttft_p99": 76.0, "e2e_p99": 328.2, "kv_waste": 98.3},
      "nanoinfer": {"throughput": 81.4, "ttft_p50": 50.1, "ttft_p99": 59.3, "e2e_p99": 319.8, "kv_waste": 25.2}
    },
    {
      "concurrency": 4,
      "naive_hf": {"throughput": 84.0, "ttft_p50": 588.7, "ttft_p99": 600.3, "e2e_p99": 853.1, "kv_waste": 98.3},
      "nanoinfer": {"throughput": 90.3, "ttft_p50": 187.5, "ttft_p99": 251.4, "e2e_p99": 1203.8, "kv_waste": 25.2}
    },
    {
      "concurrency": 8,
      "naive_hf": {"throughput": 85.2, "ttft_p50": 1288.7, "ttft_p99": 1716.8, "e2e_p99": 1968.1, "kv_waste": 98.3},
      "nanoinfer": {"throughput": 117.6, "ttft_p50": 342.0, "ttft_p99": 342.3, "e2e_p99": 1422.5, "kv_waste": 25.2}
    }
  ]
}
```

---

## CLI Reference

The server CLI supports the following configuration options:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--hf-model` | `str` | `None` | Hugging Face model repo ID or local dir (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`). |
| `--model-config` | `str` | `tiny` | Model config preset (`tiny`, `small`, `smollm2_135m`) or path to custom JSON config. |
| `--weights` | `str` | `None` | Path to weights file (`.safetensors` / `.bin` / sharded directory). |
| `--key-format` | `str` | `native` | Weight key format: `native` or `hf_llama`. |
| `--tokenizer` | `str` | `None` | Tokenizer model name or path (defaults to tiny synthetic tokenizer if not specified). |
| `--device` | `str` | `cpu` | Device for inference: `cpu`, `cuda`, or `mps`. |
| `--dtype` | `str` | `float32` | Numerical precision: `float32`, `float16`, or `bfloat16`. |
| `--num-kv-blocks` | `int` | `128` (or `512` for HF) | Number of physical KV cache blocks to allocate in memory. |
| `--block-size` | `int` | `16` | Number of tokens stored per physical KV cache block. |
| `--max-batch-size` | `int` | `32` | Maximum concurrent sequences admitted into the running batch. |
| `--host` | `str` | `0.0.0.0` | Bind host address. |
| `--port` | `int` | `8000` | Bind port number. |
| `--log-level` | `str` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

---

## Design Principles

1. **Educational & Production Aligned:** Written with clear, readable PyTorch idioms and explicit tensor shape annotations, while mirroring the exact architecture of real-world production engines (vLLM, TGI).
2. **Zero Unnecessary Dependencies:** Core server requires only `torch`, `fastapi`, `uvicorn`, `pydantic`, `httpx`, and `safetensors`.
3. **Strict Resource Safety:** Zero memory leaks, immediate reclamation on client disconnect, and automated boundary protection against sequence out-of-bounds errors.
4. **Test-First Reliability:** Every component is verified with property assertions, equivalence checks (such as cached decode vs full prefill equivalence), and end-to-end integration tests.

---

## License

This project is licensed under the [MIT License](LICENSE).
