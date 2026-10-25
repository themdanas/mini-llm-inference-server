from __future__ import annotations

import asyncio
import time
import uuid
import random
import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import httpx
from .metrics import RequestRecord, BenchmarkSummary, summarise

logger = logging.getLogger(__name__)

# Request template
@dataclass
class RequestTemplate:
    """ a single request to send to POST/generate
    prompt and max_new_tokens can be fixed or drawn from a distribution
    at runtime via the factory helpers below """

    prompt : str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = True


def _random_prompt(min_words: int = 4, max_words: int = 18) -> str:
    """Generate a dummy prompt of random length for load testing."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "model", "inference", "server", "token", "batch",
        "attention", "transformer", "layer", "weight", "gradient", "loss",
        "optimizer", "learning", "rate", "epoch", "dataset", "training",
        "inference", "generation", "sampling", "temperature", "nucleus",
    ]
    n = random.randint(min_words, max_words)
    return " ".join(random.choices(words, k=n))
 
def make_random_templates(
    n: int,
    min_words:      int = 4,
    max_words:      int = 18,
    min_new_tokens: int = 8,
    max_new_tokens: int = 48,
    temperature:    float = 1.0,
) -> List[RequestTemplate]:
    """Generate N random request templates for a load test."""
    return [
        RequestTemplate(
            prompt=_random_prompt(min_words, max_words),
            max_new_tokens=random.randint(min_new_tokens, max_new_tokens),
            temperature=temperature,
        )
        for _ in range(n)
    ]

#SSE PARSERS 
async def _consume_sse(
    response: httpx.Response,
    record: RequestRecord,
) -> None:  
    """
    Consume a Server-Sent Events response from the inference server and update the request record.
    Returns the time in seconds between the first and last token.
    """
    import json
    
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        #skip the summary "done" event
        if obj.get("object") == "done":
            record.finish_reason =obj.get("finish_reason") or record.finish_reason
            continue

        t = time.monotonic()
        if record.first_token_time is None:
            record.first_token_time = t
        record.token_times.append(t)

        if obj.get("finish_reason"):
            record.finish_reason = obj["finish_reason"]

#single request worker 
async def _send_one(
    client: httpx.AsyncClient,
    template: RequestTemplate,
    req_id: str,
    timeout: float = 120.0,
) -> RequestRecord:
    
    record = RequestRecord(
        request_id=req_id,
        submit_time=time.monotonic(),
        prompt_len=len(template.prompt.split()),   # word count as proxy
    )
 
    try:
        async with client.stream(
            "POST",
            "/generate",
            json={
                "prompt":         template.prompt,
                "max_new_tokens": template.max_new_tokens,
                "temperature":    template.temperature,
                "top_p":          template.top_p,
                "stream":         True,
                "request_id":     req_id,
            },
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                record.error = f"HTTP {resp.status_code}"
                return record
            await _consume_sse(resp, record)
 
    except httpx.TimeoutException:
        record.error = "timeout"
    except httpx.RequestError as exc:
        record.error = f"request error: {exc}"
    except Exception as exc:
        record.error = f"unexpected: {exc}"
 
    return record

# Harness
@dataclass
class HarnessConfig:
    """
    configratioin for one benchmark run
    
    concurrency
        how many request fly in parallel.
        
    """
    templates: List[RequestTemplate]
    concurrency: int = 8
    arrival_rate: float = 0.0
    timeout_s: float = 120.0
    base_url: str = "http://localhost:8000"

async def run(config: HarnessConfig) -> BenchmarkSummary:
    """ Execute a benchmark run and return aggregated statistics """

    sem      = asyncio.Semaphore(config.concurrency)
    records: List[RequestRecord] = []
    lock     = asyncio.Lock()
 
    async def _worker(template: RequestTemplate, req_id: str) -> None:
        # Poisson arrival: sleep Exponential(1/rate) before acquiring
        if config.arrival_rate > 0:
            delay = random.expovariate(config.arrival_rate)
            await asyncio.sleep(delay)
 
        async with sem:
            rec = await _send_one(client, template, req_id, config.timeout_s)
            async with lock:
                records.append(rec)
 
    async with httpx.AsyncClient(base_url=config.base_url) as client:
        wall_start = time.monotonic()
 
        tasks = [
            asyncio.create_task(
                _worker(tpl, str(uuid.uuid4())[:12])
            )
            for tpl in config.templates
        ]
        await asyncio.gather(*tasks)
 
        wall_time = time.monotonic() - wall_start
 
    summary = summarise(records, wall_time, config.concurrency)
    return summary
 

    
async def sweep(
    base_url: str = "http://localhost:8000",
    n_requests: int = 50,
    concurrency_levels: List[int] | None = None,
    max_new_tokens: int = 64,
) -> List[BenchmarkSummary]:

    """Run the harness at multiple concurrency levels and return all the summaries
       This revels the throughput-latency tradeoff curve """

    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]

    summaries = []

    for c in concurrency_levels:
        templates = make_random_templates(
            n_requests,
            max_new_tokens=max_new_tokens
        )

        config = HarnessConfig(
            templates=templates,
            concurrency=c,
            base_url=base_url,
        )

        logger.info("Sweep C=%d (%d requests) ...", c, n_requests)
        summary = await run(config)
        summaries.append(summary)
        print(summary)

    return summaries

# cli entry points
def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO)
 
    parser = argparse.ArgumentParser(description="LLM inference benchmark harness")
    parser.add_argument("--url",         default="http://localhost:8000")
    parser.add_argument("--requests",    type=int,   default=50)
    parser.add_argument("--concurrency", type=int,   default=8)
    parser.add_argument("--max-tokens",  type=int,   default=64)
    parser.add_argument("--sweep",       action="store_true",
                        help="Run a concurrency sweep instead of a single run")
    args = parser.parse_args()
 
    async def _run():
        if args.sweep:
            await sweep(
                base_url=args.url,
                n_requests=args.requests,
                max_new_tokens=args.max_tokens,
            )
        else:
            templates = make_random_templates(
                args.requests,
                max_new_tokens=args.max_tokens,
            )
            config = HarnessConfig(
                templates=templates,
                concurrency=args.concurrency,
                base_url=args.url,
            )
            summary = await run(config)
            print(summary)
 
    asyncio.run(_run())
 
 
if __name__ == "__main__":
    main()
 











