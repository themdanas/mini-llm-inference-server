# NanoInfer — A Minimal LLM Inference Server

> **Current Status**: Core model, tokenizer, and sampling modules are **complete and tested**.  
> Serving, scheduling, paged memory, and benchmarking layers are **scaffolded (empty)** — ready for implementation.

---

## Project Structure

```
mini-llm-inference-server/
├── model/              # ✅ COMPLETE — Transformer model implementation
│   ├── config.py       # ModelConfig (hyperparameters, validation, JSON I/O)
│   ├── transformer.py  # Transformer & TransformerBlock (forward + generate)
│   ├── attention.py    # MultiHeadAttention + KVCache + RoPE + GQA + FlashAttention
│   ├── layers.py       # RMSNorm, Embedding, SwiGLU/GELU FFN
│   └── weights.py      # Safe weight loading (safetensors, HF LLaMA remapping, sharded)
│
├── tokenizer/          # ✅ COMPLETE — Byte-level BPE tokenizer
│   ├── vocabulary.py   # Vocabulary (token↔id, merges, byte encoding, HF loading)
│   └── tokenizer.py    # BPETokenizer (encode/decode, batch, caching, special tokens)
│
├── sampler/            # ✅ COMPLETE — Token sampling strategies
│   ├── strategies.py   # temperature, top-k, top-p, min-p, repetition penalty, greedy/multinomial
│   └── sampler.py      # SamplerConfig (presets), LogitsProcessor pipeline, BatchSampler
│
├── serving/            # 📋 SCAFFOLDED — HTTP API (FastAPI)
│   ├── server.py       # (empty)
│   ├── engine.py       # (empty)
│   ├── routes.py       # (empty)
│   ├── schemas.py      # (empty)
│   └── streaming.py    # (empty)
│
├── scheduler/          # 📋 SCAFFOLDED — Request scheduling & continuous batching
│   ├── scheduler.py    # (empty)
│   ├── batch_manager.py# (empty)
│   ├── request.py      # (empty)
│   └── request_queue.py# (empty)
│
├── memory/             # 📋 SCAFFOLDED — Paged KV cache (vLLM-style)
│   ├── kv_cache.py     # (empty) — Note: basic KVCache lives in model/attention.py
│   ├── block_allocator.py# (empty)
│   ├── block.py        # (empty)
│   └── page_table.py   # (empty)
│
├── benchmark/          # 📋 SCAFFOLDED — Performance harness
│   ├── metrics.py      # (empty)
│   ├── harness.py      # (empty)
│   └── profiler.py     # (empty)
│
├── tests/              # ✅ COMPREHENSIVE — Unit & integration tests
│   ├── test_model.py   # 40+ tests: config, layers, RoPE, attention, KV cache, transformer
│   ├── test_tokenizer.py# BPE algo, encode/decode roundtrips, batch, special tokens
│   ├── test_sampler.py # (placeholder)
│   ├── test_scheduler.py# (placeholder)
│   ├── test_attention.py# (placeholder)
│   └── test_e2e.py     # (placeholder)
│
├── notebooks/          # (empty)
├── assets/             # (empty)
├── requirements.txt    # Runtime dependencies
├── requirements-dev.txt# Dev dependencies (pytest, etc.)
├── pyproject.toml      # (minimal)
├── .env.example        # (empty)
└── .gitignore
```

---

## ✅ What Works Today

### 1. Transformer Model (`model/`)
| Feature | Status |
|---------|--------|
| Configurable architecture (LLaMA-style) | ✅ |
| RMSNorm pre-normalization | ✅ |
| SwiGLU / GELU FFN (configurable) | ✅ |
| Rotary Positional Embeddings (RoPE) | ✅ |
| Grouped-Query Attention (GQA / MHA / MQA) | ✅ |
| KV Cache (incremental decode) | ✅ |
| Flash Attention (PyTorch 2.0 `scaled_dot_product_attention`) | ✅ |
| Weight tying (embedding ↔ lm_head) | ✅ |
| `generate()` with temperature, top-k, EOS stopping | ✅ |
| Parameter counting (`n_params()`, `n_params_approx`) | ✅ |

**Factory configs:**
```python
ModelConfig.tiny()    # 2-layer, 64-dim — for CI/unit tests
ModelConfig.small()   # 6-layer, 512-dim — ~25M params, runs on laptop GPU
```

### 2. Weight Loading (`model/weights.py`)
| Format | Support |
|--------|---------|
| `.safetensors` (single file) | ✅ |
| `.bin` / `.pt` (PyTorch pickle) | ✅ |
| Sharded checkpoints (`.index.json` + multiple shards) | ✅ |
| **HuggingFace LLaMA / Mistral / Gemma key remapping** | ✅ |
| Dtype casting on load (`float16`, `bfloat16`, `float32`) | ✅ |
| Automatic weight tying re-application after load | ✅ |

```python
from model.config import ModelConfig
from model.transformer import Transformer
from model.weights import load_weights

cfg = ModelConfig.small()
model = Transformer(cfg)
load_weights(model, "path/to/model.safetensors", key_format="hf_llama", dtype=torch.float16)
```

### 3. Tokenizer (`tokenizer/`)
| Feature | Status |
|---------|--------|
| Byte-level BPE (GPT-2 style) | ✅ |
| GPT-2 pre-tokenization regex | ✅ |
| BPE merge rules (priority-ordered) | ✅ |
| Word-level encoding cache | ✅ |
| Special tokens (BOS, EOS, PAD, UNK) | ✅ |
| Batch encoding with padding | ✅ |
| Round-trip `decode(encode(text)) == text` (ASCII) | ✅ |
| Load from HF tokenizer (`from_huggingface`) | ✅ |
| Load from `vocab.json` + `merges.txt` | ✅ |

```python
from tokenizer.tokenizer import BPETokenizer

tok = BPETokenizer.from_huggingface("meta-llama/Llama-2-7b-hf", add_bos=True)
ids = tok.encode("Hello, world!")
text = tok.decode(ids)
```

### 4. Sampler (`sampler/`)
| Feature | Status |
|---------|--------|
| Temperature scaling | ✅ |
| Top-K filtering | ✅ |
| Top-P (nucleus) filtering | ✅ |
| Min-P filtering | ✅ |
| Repetition penalty | ✅ |
| Greedy / Multinomial sampling | ✅ |
| Composable `LogitsProcessor` pipeline | ✅ |
| Per-request `SamplerConfig` | ✅ |
| Batch sampling (`BatchSampler`) | ✅ |
| Stop condition logic (EOS, max tokens, custom stop tokens) | ✅ |

**Presets:**
```python
SamplerConfig.greedy_config()    # deterministic
SamplerConfig.creative()         # high diversity
SamplerConfig.balanced()         # general chat
SamplerConfig.conservative()     # factual / coding
```

### 5. Tests (`tests/`)
- **40+ passing tests** in `test_model.py` covering:
  - Config validation, derived properties, JSON round-trip
  - RMSNorm unit RMS, learnable scale, fp16 compat
  - Embedding shape & scaling
  - SwiGLU / GELU shapes, factory
  - RoPE: frequency tables, rotation, magnitude preservation
  - Attention: output shape, causal masking, single-token decode, GQA shape
  - KV Cache: update, accumulate, overflow, reset
  - Transformer: forward shape, no NaN, weight tying, param count, determinism
  - **KV Cache Equivalence**: cached prefill/decode logits match uncached (critical)
- **BPE algorithm + tokenizer tests** in `test_tokenizer.py`

Run tests:
```bash
pytest tests/ -v
```

---

## 📋 What's Next (Implementation Roadmap)

### Phase 1: Serving Layer — **HIGH PRIORITY**
- [ ] `serving/schemas.py` — Pydantic models: `GenerateRequest`, `GenerateResponse`, `HealthResponse`
- [ ] `serving/engine.py` — Inference engine: load model+tokenizer, run `generate()`, manage KV cache
- [ ] `serving/routes.py` — FastAPI routes: `POST /generate`, `GET /health`, `GET /metrics`
- [ ] `serving/streaming.py` — SSE streaming for token-by-token output
- [ ] `serving/server.py` — Uvicorn entrypoint, CLI args, lifespan events

### Phase 2: Scheduler & Continuous Batching
- [ ] `scheduler/request.py` — Request dataclass (prompt, params, state, future)
- [ ] `scheduler/request_queue.py` — Priority queue (arrival order, priority)
- [ ] `scheduler/batch_manager.py` — Pack requests into padded batches, track per-seq state
- [ ] `scheduler/scheduler.py` — Main loop: prefill → decode steps, evict finished, admit new

### Phase 3: Paged KV Cache (vLLM-style)
- [ ] `memory/block.py` — Fixed-size KV block (e.g., 16 tokens)
- [ ] `memory/block_allocator.py` — Free list, allocate/free blocks
- [ ] `memory/page_table.py` — Per-request logical→physical block mapping
- [ ] `memory/kv_cache.py` — PagedKVCache using above; integrate with `MultiHeadAttention`

### Phase 4: Benchmarks & Observability
- [ ] `benchmark/harness.py` — Throughput/latency sweeps (batch size, seq len, concurrency)
- [ ] `benchmark/metrics.py` — TTFT, ITL, tokens/sec, memory usage
- [ ] `benchmark/profiler.py` — PyTorch profiler integration, NVTX markers

### Phase 5: Polish
- [ ] Fill in `test_sampler.py`, `test_scheduler.py`, `test_e2e.py`
- [ ] Add example notebooks in `notebooks/`
- [ ] Dockerfile, docker-compose.yml
- [ ] CI/CD (GitHub Actions)

---

## Quick Start

```bash
# 1. Install PyTorch first (match your hardware)
# CUDA 12.1:
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
# Apple Silicon (MPS):
pip install torch==2.3.1

# 2. Install NanoInfer deps
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# 4. Quick sanity check (Python REPL)
python -c "
from model.config import ModelConfig
from model.transformer import Transformer
from tokenizer.tokenizer import BPETokenizer

cfg = ModelConfig.tiny()
model = Transformer(cfg)
tok = BPETokenizer.tiny_test()

ids = tok.encode('hello')
print('Encoded:', ids)

out = model.generate(torch.tensor([ids]), max_new_tokens=5)
print('Generated:', tok.decode(out[0].tolist()))
"
```

---

## Design Philosophy

| Principle | How It's Applied |
|-----------|------------------|
| **Minimal dependencies** | Only PyTorch, FastAPI, tokenizers, safetensors, numpy, pyyaml |
| **No pickle** | `safetensors` for weights; `weights_only=True` for `.bin` |
| **Readable over clever** | Plain PyTorch, minimal `einops`, explicit shapes in comments |
| **Test-first** | Every core component has unit tests; KV cache equivalence is mandatory |
| **HF compatible** | Load LLaMA/Mistral/Gemma checkpoints via key remapping |
| **Production-ready patterns** | Paged attention, continuous batching, SSE streaming (planned) |

---

## License

MIT — see `LICENSE` (to be added).

---

## Contributing

1. Pick an empty module from the roadmap above
2. Write tests first (`tests/test_<module>.py`)
3. Implement the module
4. Run `pytest tests/ -v` — all must pass
5. Open a PR

---

*Built with ❤️ for learning and lightweight inference.*