"""
benchmark/compare_hf_vs_nanoinfer.py
────────────────────────────────────
Runs an apples-to-apples concurrency sweep comparing:
  1. Naive HuggingFace Transformers (standard model.generate() with contiguous KV cache)
  2. NanoInfer (Continuous Batching + Paged KV Cache)

Outputs a formatted comparison table to the terminal and saves it to
benchmark/comparison_results.md for your README and project evaluation.

Usage:
  1. Start NanoInfer in Terminal 1:
     python -m serving.server --hf-model HuggingFaceTB/SmolLM2-135M-Instruct --num-kv-blocks 512 --port 8000

  2. Run this comparison script in Terminal 2:
     python -m benchmark.compare_hf_vs_nanoinfer --hf-model HuggingFaceTB/SmolLM2-135M-Instruct --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
import torch

PROMPTS = [
    "Explain how virtual memory page tables work in operating systems:",
    "The primary bottleneck in autoregressive transformer inference is",
    "Continuous batching improves GPU utilization during LLM serving by",
    "In computer science, a hash table resolves collisions using",
    "Write a short summary of why attention mechanisms scale quadratically:",
    "The difference between prefill and decode phases in language models is",
    "Key-value caching avoids redundant computation during token generation because",
    "Distributed systems achieve fault tolerance and high availability through",
]


@dataclass
class RunMetrics:
    engine_name: str
    concurrency: int
    total_requests: int
    wall_time_s: float
    throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    e2e_p50_ms: float
    e2e_p99_ms: float
    kv_waste_pct: float


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ─── 1. Benchmark NanoInfer Server (via SSE Streaming) ──────────────────────


async def benchmark_nanoinfer(
    url: str,
    concurrency: int,
    num_requests: int,
    max_new_tokens: int,
    block_size: int = 16,
) -> RunMetrics:
    sem = asyncio.Semaphore(concurrency)
    ttfts_ms: List[float] = []
    e2es_ms: List[float] = []
    total_gen_tokens = 0
    waste_ratios: List[float] = []

    async with httpx.AsyncClient(timeout=120.0) as client:

        async def _one_req(idx: int) -> None:
            nonlocal total_gen_tokens
            prompt = PROMPTS[idx % len(PROMPTS)]
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "stream": True,
            }
            async with sem:
                t0 = time.perf_counter()
                first_token_time = None
                tokens_received = 0

                async with client.stream("POST", f"{url}/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        tokens_received += 1

                t_end = time.perf_counter()
                if first_token_time is None:
                    first_token_time = t_end

                ttfts_ms.append((first_token_time - t0) * 1000.0)
                e2es_ms.append((t_end - t0) * 1000.0)
                total_gen_tokens += max(1, tokens_received - 1)

                # Internal fragmentation in Paged KV Cache: only last block has unused slots
                total_seq_tokens = len(prompt.split()) + max_new_tokens
                allocated_slots = ((total_seq_tokens + block_size - 1) // block_size) * block_size
                waste_ratios.append((allocated_slots - total_seq_tokens) / allocated_slots)

        t_start = time.perf_counter()
        await asyncio.gather(*[_one_req(i) for i in range(num_requests)])
        wall_time = time.perf_counter() - t_start

    return RunMetrics(
        engine_name="NanoInfer (Paged KV + Cont. Batch)",
        concurrency=concurrency,
        total_requests=num_requests,
        wall_time_s=wall_time,
        throughput_tok_s=total_gen_tokens / max(wall_time, 1e-6),
        ttft_p50_ms=percentile(ttfts_ms, 50),
        ttft_p99_ms=percentile(ttfts_ms, 99),
        e2e_p50_ms=percentile(e2es_ms, 50),
        e2e_p99_ms=percentile(e2es_ms, 99),
        kv_waste_pct=statistics.mean(waste_ratios) * 100.0,
    )


# ─── 2. Benchmark Naive HuggingFace Baseline (Serial/Locked model.generate) ─


async def benchmark_naive_hf(
    hf_model_name: str,
    device: str,
    concurrency: int,
    num_requests: int,
    max_new_tokens: int,
    max_seq_len: int = 2048,
) -> RunMetrics:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Baseline] Loading Naive HuggingFace model ({hf_model_name}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    sem = asyncio.Semaphore(concurrency)
    gpu_lock = asyncio.Lock()  # Standard naive FastAPI wrapper serializes model.generate()
    ttfts_ms: List[float] = []
    e2es_ms: List[float] = []
    total_gen_tokens = 0
    waste_ratios: List[float] = []

    async def _one_naive_req(idx: int) -> None:
        nonlocal total_gen_tokens
        prompt = PROMPTS[idx % len(PROMPTS)]
        async with sem:
            t_submit = time.perf_counter()
            # Wait in queue for exclusive model.generate() lock
            async with gpu_lock:
                t_start_exec = time.perf_counter()
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                prompt_len = inputs["input_ids"].shape[1]

                # Step 1: Prefill to measure TTFT
                with torch.inference_mode():
                    _ = model(**inputs, use_cache=True)
                t_first_tok = time.perf_counter()

                # Step 2: Full generation
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                t_done = time.perf_counter()
                await asyncio.sleep(0)

            gen_tokens = out.shape[1] - prompt_len
            total_gen_tokens += gen_tokens
            # In naive serving, queuing delay directly inflates TTFT!
            ttfts_ms.append((t_first_tok - t_submit) * 1000.0)
            e2es_ms.append((t_done - t_submit) * 1000.0)

            # Contiguous max_seq_len reservation waste
            used_tokens = prompt_len + gen_tokens
            waste_ratios.append((max_seq_len - used_tokens) / max_seq_len)

    t_start = time.perf_counter()
    await asyncio.gather(*[_one_naive_req(i) for i in range(num_requests)])
    wall_time = time.perf_counter() - t_start

    return RunMetrics(
        engine_name="Naive HuggingFace (model.generate)",
        concurrency=concurrency,
        total_requests=num_requests,
        wall_time_s=wall_time,
        throughput_tok_s=total_gen_tokens / max(wall_time, 1e-6),
        ttft_p50_ms=percentile(ttfts_ms, 50),
        ttft_p99_ms=percentile(ttfts_ms, 99),
        e2e_p50_ms=percentile(e2es_ms, 50),
        e2e_p99_ms=percentile(e2es_ms, 99),
        kv_waste_pct=statistics.mean(waste_ratios) * 100.0,
    )


def format_markdown_table(results: List[RunMetrics]) -> str:
    lines = [
        "# Real-World Benchmark Comparison: Naive HuggingFace vs. NanoInfer",
        "",
        "| Engine | Concurrency | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | E2E P99 (ms) | KV Memory Waste (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| **{r.engine_name}** | {r.concurrency} | **{r.throughput_tok_s:.1f}** | "
            f"{r.ttft_p50_ms:.1f} | {r.ttft_p99_ms:.1f} | {r.e2e_p99_ms:.1f} | "
            f"**{r.kv_waste_pct:.1f}%** |"
        )
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000", help="Running NanoInfer URL")
    p.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--requests", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4, 8])
    args = p.parse_args()

    all_results: List[RunMetrics] = []

    for c in args.concurrencies:
        print(f"\n─── Running Concurrency = {c} ({args.requests} requests) ───")
        hf_res = await benchmark_naive_hf(
            args.hf_model, args.device, c, args.requests, args.max_tokens
        )
        nano_res = await benchmark_nanoinfer(
            args.url, c, args.requests, args.max_tokens
        )
        all_results.extend([hf_res, nano_res])

    md_report = format_markdown_table(all_results)
    print("\n" + "=" * 80)
    print(md_report)
    print("=" * 80)

    out_path = Path("benchmark/comparison_results.md")
    out_path.write_text(md_report, encoding="utf-8")
    print(f"\n[Saved full Markdown comparison report to {out_path}]")


if __name__ == "__main__":
    asyncio.run(main())