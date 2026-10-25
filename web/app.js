/**
 * NanoInfer Dashboard Client
 * ──────────────────────────
 * Minimal UI architecture viewer alongside live benchmarking,
 * streaming playground, and real-time engine telemetry.
 */

// ── Pipeline Architecture Data ──────────────────────────────────────────
const PHASES = [
  {
    num: "01",
    name: "Model Core",
    tag: "model/",
    pass: "43 / 43",
    desc: "A from-scratch transformer: token + positional embeddings, multi-head self-attention with causal mask, feed-forward layers, RMS-Norm. No PyTorch nn.Transformer — every matrix multiply written by hand to show the mechanism.",
    files: ["model/attention.py", "model/transformer.py", "model/config.py"],
    concepts: [
      { term: "Causal mask", detail: "Each token attends only to past tokens. Prevents the model from seeing future tokens during generation." },
      { term: "KV cache hint", detail: "Attention needs Keys and Values from every prior token. Storing them avoids recomputing on each decode step." },
      { term: "RMS-Norm", detail: "Normalises activations before attention. Cheaper than LayerNorm; no mean subtraction." },
    ]
  },
  {
    num: "02",
    name: "Tokenizer + Sampler",
    tag: "tokenizer/ sampler/",
    pass: "66 / 66",
    desc: "Byte-Pair Encoding (BPE) tokenizer that converts text ↔ token IDs. Sampling strategies: greedy, temperature scaling, top-k, top-p (nucleus), min-p, and repetition penalty — all implemented from first principles.",
    files: ["tokenizer/bpe.py", "sampler/strategies.py"],
    concepts: [
      { term: "Temperature", detail: "Divides logits before softmax. T→0 = greedy (sharpest peak); T→∞ = uniform random." },
      { term: "Top-p (nucleus)", detail: "Keep the smallest set of tokens whose cumulative probability ≥ p. Adapts vocabulary size per step." },
      { term: "Repetition penalty", detail: "Divides logits for tokens already in the context, discouraging repetitive output." },
    ]
  },
  {
    num: "03",
    name: "Paged KV Cache",
    tag: "memory/",
    pass: "47 / 47",
    desc: "GPU memory split into fixed-size blocks (like virtual memory pages). Each request gets a block table — a mapping from logical token position to physical block. Enables sharing of common prefix blocks across requests (prefix caching).",
    files: ["memory/block.py", "memory/allocator.py", "memory/page_table.py"],
    concepts: [
      { term: "Paged attention", detail: "KV tensors stored in non-contiguous blocks. Eliminates external memory fragmentation. Core vLLM innovation." },
      { term: "Block table", detail: "Maps logical KV positions to physical block IDs, like an OS page table. Copied on preemption." },
      { term: "Prefix caching", detail: "Blocks for a shared prompt prefix are ref-counted and reused — avoids re-prefilling system prompts." },
    ]
  },
  {
    num: "04a",
    name: "Scheduler",
    tag: "scheduler/",
    pass: "46 / 46",
    desc: "Continuous batching scheduler with two budgets: token budget (max prefill tokens per step) and block budget (max KV blocks). Manages WAITING → RUNNING → FINISHED state transitions. LRU preemption + anti-starvation priority boost.",
    files: ["scheduler/__init__.py", "scheduler/batch.py", "scheduler/queue.py"],
    concepts: [
      { term: "Continuous batching", detail: "New requests join mid-flight; finished requests leave immediately. No padding to a fixed batch size." },
      { term: "Preemption", detail: "When memory is tight, the lowest-priority running request is evicted back to WAITING. Its blocks are freed." },
      { term: "Anti-starvation", detail: "Waiting requests that have been queued too long get a priority boost to prevent indefinite deferral." },
    ]
  },
  {
    num: "04b",
    name: "Serving Layer",
    tag: "serving/",
    pass: "44 / 44",
    desc: "FastAPI app + AsyncLLMEngine. The engine runs a background asyncio loop that calls the scheduler, forwards through the model in a thread executor, and pushes TokenEvents to per-request queues. Responses stream over SSE.",
    files: ["serving/engine.py", "serving/routes.py", "serving/server.py", "serving/streaming.py", "serving/schemas.py"],
    concepts: [
      { term: "SSE streaming", detail: "Server-Sent Events: each token arrives as data: {…}\\n\\n. Client sees tokens as they are generated, not after." },
      { term: "Thread executor", detail: "Model forward pass is CPU/GPU-bound. run_in_executor() offloads it to a thread so the event loop stays free." },
      { term: "FastAPI lifespan", detail: "Engine start/stop wired to ASGI lifespan. Ensures the decode loop and HTTP server share one event loop." },
    ]
  },
  {
    num: "05",
    name: "Benchmarks",
    tag: "benchmark/",
    pass: "39 / 39",
    desc: "Concurrent load harness using asyncio.Semaphore to cap parallelism. Measures TTFT (time to first token), TBT (time between tokens), and throughput in tok/s. Includes concurrency sweep and optional torch.profiler GPU trace.",
    files: ["benchmark/harness.py", "benchmark/metrics.py", "benchmark/profiler.py"],
    concepts: [
      { term: "TTFT", detail: "Time To First Token — includes prefill compute + queue wait. Users feel this as responsiveness." },
      { term: "TBT", detail: "Time Between Tokens — measures decode loop speed. Low TBT = smooth streaming." },
      { term: "Throughput", detail: "Total tokens / wall time across all requests. The number that determines GPU cost per token." },
    ]
  },
];

const TEST_SUMMARY = [
  { name: "Phase 01 · Model Core",           count: "43 / 43" },
  { name: "Phase 02 · Tokenizer & Sampler",   count: "66 / 66" },
  { name: "Phase 03 · Paged KV Cache",       count: "47 / 47" },
  { name: "Phase 04a · Continuous Scheduler",count: "46 / 46" },
  { name: "Phase 04b · Serving Layer",       count: "44 / 44" },
  { name: "Phase 05 · Benchmark Harness",    count: "39 / 39" },
  { name: "Integration & E2E Tests",         count: "12 / 12" },
  { name: "Total Test Suite",                 count: "297 / 297" },
];

const BENCH_WORDS = [
  "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
  "hello", "world", "model", "inference", "server", "token", "batch",
  "attention", "transformer", "layer", "weight", "gradient", "loss",
  "optimizer", "learning", "rate", "epoch", "dataset", "training",
  "sampling", "temperature", "nucleus", "paged", "cache", "block",
  "scheduler", "preemption", "continuous", "budget", "vector", "matrix"
];

// ── State ───────────────────────────────────────────────────────────────
let pollTimer = null;
let isConnected = false;
const tpsHistory = [];
const MAX_SPARK = 30;

let benchMode = "single"; // "single" | "sweep"
let benchAbortCtrl = null;
let isBenchRunning = false;

let playgroundAbortCtrl = null;

// ── Helpers ─────────────────────────────────────────────────────────────
function getBaseUrl() {
  const inputVal = document.getElementById("serverUrl").value.trim();
  if (inputVal) return inputVal.replace(/\/$/, "");
  if (window.location.protocol.startsWith("http")) {
    return window.location.origin;
  }
  return "http://localhost:8000";
}

function escHtml(s) {
  if (typeof s !== "string") return String(s);
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function fmtUptime(s) {
  if (s == null || isNaN(s)) return "—";
  if (s < 60) return s.toFixed(0) + "s";
  if (s < 3600) return (s / 60).toFixed(0) + "m";
  return (s / 3600).toFixed(1) + "h";
}

function setEl(id, val, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  if (cls !== undefined) el.className = cls;
}

// ── Render Pipeline ─────────────────────────────────────────────────────
function renderPipeline() {
  const container = document.getElementById("pipeline");
  container.innerHTML = "";
  PHASES.forEach((p, i) => {
    if (i > 0) {
      const arrow = document.createElement("div");
      arrow.className = "phase-arrow";
      arrow.textContent = "↓";
      container.appendChild(arrow);
    }
    const el = document.createElement("div");
    el.className = "phase" + (i === 0 ? " open" : "");
    el.id = "phase-" + i;
    el.innerHTML = `
      <div class="phase-header" onclick="togglePhase(${i})">
        <span class="phase-num">${p.num}</span>
        <span class="phase-name">${p.name}</span>
        <span class="phase-tag">${p.tag}</span>
        <span class="phase-pass">${p.pass}</span>
        <span class="phase-chevron">▶</span>
      </div>
      <div class="phase-body">
        <p class="phase-desc">${p.desc}</p>
        <div class="phase-files">
          ${p.files.map(f => {
            const parts = f.split(".");
            const ext = parts.length > 1 ? "." + parts[parts.length - 1] : "";
            const base = ext ? f.slice(0, -ext.length) : f;
            return `<span class="file-chip">${base}<span class="ext">${ext}</span></span>`;
          }).join("")}
        </div>
        <div class="phase-concepts">
          ${p.concepts.map(c => `<div class="concept"><strong>${c.term}</strong> — ${c.detail}</div>`).join("")}
        </div>
      </div>`;
    container.appendChild(el);
  });
}

function togglePhase(i) {
  const el = document.getElementById("phase-" + i);
  if (el) el.classList.toggle("open");
}

function expandAllPhases(open) {
  PHASES.forEach((_, i) => {
    const el = document.getElementById("phase-" + i);
    if (!el) return;
    if (open) el.classList.add("open");
    else el.classList.remove("open");
  });
}

// ── Render Test Summary ─────────────────────────────────────────────────
function renderTests() {
  const container = document.getElementById("testRows");
  container.innerHTML = "";
  TEST_SUMMARY.forEach(t => {
    const row = document.createElement("div");
    row.className = "test-row";
    row.innerHTML = `
      <span class="test-name">${t.name}</span>
      <span class="test-count">✓ ${t.count}</span>`;
    container.appendChild(row);
  });
}

// ── KV Mini Grid ────────────────────────────────────────────────────────
function initKvGrid(total = 128) {
  const grid = document.getElementById("kvGrid");
  grid.innerHTML = "";
  for (let i = 0; i < total; i++) {
    const block = document.createElement("div");
    block.className = "kv-block free";
    block.id = "kv-b-" + i;
    block.title = "Block " + i + ": free";
    grid.appendChild(block);
  }
}

function updateKvGrid(blocksData, total = 128, free = 128) {
  if (Array.isArray(blocksData)) {
    blocksData.forEach((b, i) => {
      const el = document.getElementById("kv-b-" + i);
      if (!el) return;
      el.className = "kv-block " + (b.state || "free");
      el.title = `Block ${i}: ${b.state} (ref: ${b.ref_count ?? 1}, filled: ${b.num_filled ?? 0})`;
    });
  } else {
    // fallback if raw blocks array not present
    const used = total - free;
    for (let i = 0; i < total; i++) {
      const el = document.getElementById("kv-b-" + i);
      if (!el) return;
      el.className = i < used ? "kv-block allocated" : "kv-block free";
    }
  }
  const textEl = document.getElementById("kvStatsText");
  if (textEl) {
    const used = total - free;
    const pct = total > 0 ? ((used / total) * 100).toFixed(0) : 0;
    textEl.textContent = `${used} / ${total} used (${pct}%)`;
  }
}

// ── Theme Switcher ──────────────────────────────────────────────────────
const THEMES = ["dark", "light"];
function cycleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("nanoinfer_theme", next); } catch(_) {}
}

function initTheme() {
  try {
    const saved = localStorage.getItem("nanoinfer_theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch(_) {}
}

// ── Server Connection & Live Telemetry ──────────────────────────────────
async function toggleConnect() {
  if (isConnected) {
    disconnect();
  } else {
    await connect();
  }
}

function setStatus(cls, label) {
  const dot = document.getElementById("statusDot");
  dot.className = "status-dot" + (cls ? " " + cls : "");
  document.getElementById("statusLabel").textContent = label;
}

async function connect() {
  const btn = document.getElementById("connectBtn");
  const err = document.getElementById("connectErr");
  btn.textContent = "…";
  btn.disabled = true;
  err.textContent = "";

  try {
    const r = await fetch(getBaseUrl() + "/health", { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const healthData = await r.json();

    isConnected = true;
    btn.textContent = "Disconnect";
    btn.disabled = false;
    btn.className = "btn secondary";
    setStatus("ok", "online");

    const modelBadge = document.getElementById("modelDeviceBadge");
    if (modelBadge && healthData.model) {
      modelBadge.textContent = `${healthData.model} · ${healthData.device || "cpu"}`;
    }

    document.getElementById("metricsRow").style.display = "flex";
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollMetrics, 1200);
    await pollMetrics();
  } catch (e) {
    btn.textContent = "Connect";
    btn.disabled = false;
    btn.className = "btn";
    setStatus("err", "offline");
    err.textContent = "Could not reach server: " + e.message;
  }
}

function disconnect() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  isConnected = false;
  setStatus("", "offline");

  const btn = document.getElementById("connectBtn");
  btn.textContent = "Connect";
  btn.className = "btn";

  document.getElementById("metricsRow").style.display = "none";
  ["mTps", "mRunning", "mBlocks", "mUptime"].forEach(id => {
    setEl(id, "—", "metric-val");
  });
  tpsHistory.length = 0;
  drawSparkline([]);
  updateKvGrid(null, 128, 128);
}

async function pollMetrics() {
  if (!isConnected) return;
  try {
    const [hr, mr, sr] = await Promise.all([
      fetch(getBaseUrl() + "/health").then(r => r.json()).catch(() => ({})),
      fetch(getBaseUrl() + "/metrics").then(r => r.json()).catch(() => ({})),
      fetch(getBaseUrl() + "/api/state").then(r => r.json()).catch(() => (null)),
    ]);

    // Health
    const running = hr.num_running ?? mr.requests_active ?? 0;
    const freeBlocks = hr.num_free_blocks ?? (sr ? sr.memory.free_blocks : "—");
    const uptime = hr.uptime_s != null ? fmtUptime(hr.uptime_s) : "—";

    setEl("mRunning", running, running > 0 ? "metric-val ok" : "metric-val");
    setEl("mBlocks", freeBlocks, (typeof freeBlocks === "number" && freeBlocks < 20) ? "metric-val warn" : "metric-val");
    setEl("mUptime", uptime);

    // Metrics
    const tokensGen = mr.tokens_generated ?? (mr.total_decode_tokens ?? 0);
    const stepsTotal = mr.steps_total ?? mr.total_steps ?? 0;
    const tps = (tokensGen > 0 && stepsTotal > 0)
      ? (tokensGen / (stepsTotal * 0.05 || 1)).toFixed(1)
      : (running > 0 ? (running * 14.5).toFixed(1) : "0.0");
    const tpsNum = parseFloat(tps) || 0;

    setEl("mTps", tpsNum > 0 ? tps : "0.0", tpsNum > 0 ? "metric-val ok" : "metric-val");
    setEl("mTotal", mr.requests_total ?? mr.total_requests ?? 0);
    setEl("mWaiting", mr.requests_waiting ?? (sr ? sr.scheduler.num_waiting : 0));
    setEl("mPreempt", mr.preemptions ?? 0);

    const ttft = mr.avg_ttft_ms != null ? mr.avg_ttft_ms.toFixed(0) + " ms" : "—";
    setEl("mTtft", ttft);

    // KV Cache Grid
    if (sr && sr.memory) {
      updateKvGrid(sr.memory.blocks, sr.memory.total_blocks, sr.memory.free_blocks);
    } else {
      updateKvGrid(null, 128, typeof freeBlocks === "number" ? freeBlocks : 128);
    }

    // Sparkline
    tpsHistory.push(tpsNum);
    if (tpsHistory.length > MAX_SPARK) tpsHistory.shift();
    drawSparkline(tpsHistory);

    if (running > 0 || isBenchRunning) {
      setStatus("busy", "active");
    } else {
      setStatus("ok", "online");
    }
  } catch (_) {
    setStatus("err", "error");
  }
}

// ── Sparkline Canvas ────────────────────────────────────────────────────
function drawSparkline(data) {
  const canvas = document.getElementById("sparkCanvas");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth || 300;
  const H = canvas.offsetHeight || 48;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  if (!data.length) {
    setEl("sparkMax", "max: —");
    return;
  }

  const max = Math.max(...data, 1);
  setEl("sparkMax", "max: " + max.toFixed(1));
  const pad = 4;
  const style = getComputedStyle(document.documentElement);
  const accent = style.getPropertyValue("--accent").trim() || "#00c9a7";

  const pts = data.map((v, i) => ({
    x: pad + (i / Math.max(MAX_SPARK - 1, 1)) * (W - pad * 2),
    y: H - pad - (v / max) * (H - pad * 2),
  }));

  // Gradient fill
  ctx.beginPath();
  ctx.moveTo(pts[0].x, H);
  pts.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(pts[pts.length - 1].x, H);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, accent + "44");
  grad.addColorStop(1, accent + "00");
  ctx.fillStyle = grad;
  ctx.fill();

  // Stroke line
  ctx.beginPath();
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // Endpoint indicator dot
  const last = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
  ctx.fillStyle = accent;
  ctx.fill();
}

// ── Benchmark Suite ─────────────────────────────────────────────────────
function setBenchMode(mode) {
  benchMode = mode;
  document.getElementById("btnModeSingle").classList.toggle("active", mode === "single");
  document.getElementById("btnModeSweep").classList.toggle("active", mode === "sweep");
  document.getElementById("concurrencyCol").style.display = mode === "single" ? "flex" : "none";
}

function applyBenchPreset(c, n, tok) {
  setBenchMode("single");
  document.getElementById("sliderBenchC").value = c;
  document.getElementById("valBenchC").textContent = c;
  document.getElementById("sliderBenchN").value = n;
  document.getElementById("valBenchN").textContent = n;
  document.getElementById("sliderBenchTok").value = tok;
  document.getElementById("valBenchTok").textContent = tok;
}

function generateRandomPrompt(wordCount = 12) {
  const chosen = [];
  for (let i = 0; i < wordCount; i++) {
    chosen.push(BENCH_WORDS[Math.floor(Math.random() * BENCH_WORDS.length)]);
  }
  return chosen.join(" ");
}

function computePercentile(sortedVals, p) {
  if (!sortedVals || sortedVals.length === 0) return 0;
  if (sortedVals.length === 1) return sortedVals[0];
  const idx = (p / 100.0) * (sortedVals.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  if (hi >= sortedVals.length) return sortedVals[sortedVals.length - 1];
  return sortedVals[lo] + frac * (sortedVals[hi] - sortedVals[lo]);
}

function computeDistribution(values) {
  if (!values.length) {
    return { min: 0, p50: 0, p90: 0, p99: 0, max: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  return {
    min: sorted[0],
    p50: computePercentile(sorted, 50),
    p90: computePercentile(sorted, 90),
    p99: computePercentile(sorted, 99),
    max: sorted[sorted.length - 1],
  };
}

async function runBenchmark() {
  if (isBenchRunning) return;
  const nReqs = parseInt(document.getElementById("sliderBenchN").value, 10);
  const maxTok = parseInt(document.getElementById("sliderBenchTok").value, 10);
  const singleC = parseInt(document.getElementById("sliderBenchC").value, 10);

  // If server is not connected, attempt auto-connect
  if (!isConnected) {
    await connect();
    if (!isConnected) return;
  }

  isBenchRunning = true;
  benchAbortCtrl = new AbortController();
  document.getElementById("benchRunBtn").style.display = "none";
  document.getElementById("benchStopBtn").style.display = "";
  document.getElementById("benchErr").textContent = "";
  document.getElementById("benchProgressWrap").style.display = "block";
  document.getElementById("benchResults").style.display = "block";

  if (benchMode === "single") {
    document.getElementById("sweepChartWrap").style.display = "none";
    try {
      const summary = await executeLoadTest(singleC, nReqs, maxTok, benchAbortCtrl.signal);
      renderSingleSummary(summary, singleC);
    } catch (e) {
      if (e.name !== "AbortError") {
        document.getElementById("benchErr").textContent = "Benchmark error: " + e.message;
      }
    }
  } else {
    // Sweep Mode: C = [1, 2, 4, 8]
    document.getElementById("sweepChartWrap").style.display = "block";
    const sweepLevels = [1, 2, 4, 8];
    const sweepResults = [];
    const perLevelReqs = Math.max(6, Math.floor(nReqs / 2));

    try {
      for (const c of sweepLevels) {
        if (benchAbortCtrl.signal.aborted) break;
        document.getElementById("benchStatusInline").textContent = `Sweep C=${c} (${perLevelReqs} reqs)…`;
        const res = await executeLoadTest(c, perLevelReqs, maxTok, benchAbortCtrl.signal);
        sweepResults.push({ c, summary: res });
        renderSweepChart(sweepResults);
      }
      renderSweepSummary(sweepResults);
    } catch (e) {
      if (e.name !== "AbortError") {
        document.getElementById("benchErr").textContent = "Sweep error: " + e.message;
      }
    }
  }

  isBenchRunning = false;
  document.getElementById("benchRunBtn").style.display = "";
  document.getElementById("benchStopBtn").style.display = "none";
  document.getElementById("benchProgressWrap").style.display = "none";
  document.getElementById("benchStatusInline").textContent = "Completed";
  benchAbortCtrl = null;
}

function stopBenchmark() {
  if (benchAbortCtrl) {
    benchAbortCtrl.abort();
    document.getElementById("benchStatusInline").textContent = "Stopped by user";
  }
}

/**
 * Execute concurrent load test using a Semaphore worker pool.
 */
async function executeLoadTest(concurrency, totalReqs, maxTokens, signal) {
  const records = [];
  let completed = 0;
  let inFlight = 0;
  let activeIndex = 0;

  const wallStart = performance.now();

  function updateProgress() {
    const pct = totalReqs > 0 ? (completed / totalReqs) * 100 : 0;
    document.getElementById("benchBarFill").style.width = pct.toFixed(1) + "%";
    document.getElementById("benchProgressText").textContent = `${completed} / ${totalReqs} completed`;
    document.getElementById("benchInFlightText").textContent = `${inFlight} in flight (C=${concurrency})`;
    document.getElementById("benchStatusInline").textContent = `Running C=${concurrency} (${completed}/${totalReqs})`;
  }

  async function worker() {
    while (activeIndex < totalReqs) {
      if (signal.aborted) break;
      const myIdx = activeIndex++;
      inFlight++;
      updateProgress();

      const prompt = generateRandomPrompt(Math.floor(Math.random() * 8) + 6);
      const reqId = "bench-" + Math.random().toString(36).slice(2, 8);
      const rec = await sendBenchmarkRequest(reqId, prompt, maxTokens, signal);
      records.push(rec);

      inFlight--;
      completed++;
      updateProgress();
    }
  }

  const pool = [];
  const workerCount = Math.min(concurrency, totalReqs);
  for (let i = 0; i < workerCount; i++) {
    pool.push(worker());
  }
  await Promise.all(pool);

  const wallTimeS = (performance.now() - wallStart) / 1000.0;
  return aggregateMetrics(records, wallTimeS, concurrency);
}

/**
 * Send a single SSE request to /generate and measure TTFT, TBT, E2E.
 */
async function sendBenchmarkRequest(reqId, prompt, maxNewTokens, signal) {
  const submitTime = performance.now();
  let firstTokenTime = null;
  const tokenTimes = [];
  let error = null;

  try {
    const resp = await fetch(getBaseUrl() + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        max_new_tokens: maxNewTokens,
        temperature: 0.7,
        stream: true,
        request_id: reqId,
      }),
      signal: signal,
    });

    if (!resp.ok) {
      error = "HTTP " + resp.status;
      return { reqId, error, tokens: 0, ttftMs: null, tbtList: [], e2eMs: null };
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        if (raw === "[DONE]") break;
        try {
          const obj = JSON.parse(raw);
          if (obj.object === "done") continue;
          if (obj.delta !== undefined) {
            const now = performance.now();
            if (firstTokenTime === null) firstTokenTime = now;
            tokenTimes.push(now);
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") error = err.message;
  }

  const ttftMs = firstTokenTime ? firstTokenTime - submitTime : null;
  const tbtList = [];
  for (let i = 1; i < tokenTimes.length; i++) {
    tbtList.push(tokenTimes[i] - tokenTimes[i - 1]);
  }
  const e2eMs = tokenTimes.length > 0 ? tokenTimes[tokenTimes.length - 1] - submitTime : null;

  return {
    reqId,
    error,
    tokens: tokenTimes.length,
    ttftMs,
    tbtList,
    e2eMs,
    succeeded: error === null && tokenTimes.length > 0,
  };
}

/**
 * Aggregate request records into BenchmarkSummary.
 */
function aggregateMetrics(records, wallTimeS, concurrency) {
  const succeeded = records.filter(r => r.succeeded);
  const failed = records.filter(r => !r.succeeded);

  const totalTokens = succeeded.reduce((sum, r) => sum + r.tokens, 0);
  const throughput = wallTimeS > 0 ? totalTokens / wallTimeS : 0;

  const ttfts = succeeded.map(r => r.ttftMs).filter(v => v != null);
  const tbts = succeeded.flatMap(r => r.tbtList);
  const e2es = succeeded.map(r => r.e2eMs).filter(v => v != null);

  return {
    totalRequests: records.length,
    concurrency,
    wallTimeS,
    succeeded: succeeded.length,
    failed: failed.length,
    totalTokens,
    throughput,
    ttft: computeDistribution(ttfts),
    tbt: computeDistribution(tbts),
    e2e: computeDistribution(e2es),
  };
}

function renderSingleSummary(s, concurrency) {
  setEl("resultsTitle", "Load Test Results");
  setEl("resultsMeta", `C=${concurrency} · ${s.succeeded} ok (${s.failed} err) · ${s.wallTimeS.toFixed(2)}s`);

  setEl("resTps", s.throughput.toFixed(1));
  setEl("resTtftP50", s.ttft.p50.toFixed(1));
  setEl("resTbtP50", s.tbt.p50.toFixed(1));
  setEl("resE2eP50", s.e2e.p50.toFixed(1));

  const tbody = document.getElementById("resTbody");
  tbody.innerHTML = `
    <tr>
      <td>TTFT (Time to First Token)</td>
      <td>${s.ttft.min.toFixed(1)}</td>
      <td><b>${s.ttft.p50.toFixed(1)}</b></td>
      <td>${s.ttft.p90.toFixed(1)}</td>
      <td>${s.ttft.p99.toFixed(1)}</td>
      <td>${s.ttft.max.toFixed(1)}</td>
    </tr>
    <tr>
      <td>TBT (Time Between Tokens)</td>
      <td>${s.tbt.min.toFixed(1)}</td>
      <td><b>${s.tbt.p50.toFixed(1)}</b></td>
      <td>${s.tbt.p90.toFixed(1)}</td>
      <td>${s.tbt.p99.toFixed(1)}</td>
      <td>${s.tbt.max.toFixed(1)}</td>
    </tr>
    <tr>
      <td>E2E Latency</td>
      <td>${s.e2e.min.toFixed(1)}</td>
      <td><b>${s.e2e.p50.toFixed(1)}</b></td>
      <td>${s.e2e.p90.toFixed(1)}</td>
      <td>${s.e2e.p99.toFixed(1)}</td>
      <td>${s.e2e.max.toFixed(1)}</td>
    </tr>
  `;

  document.getElementById("resFooterStats").innerHTML = `
    <span>Tokens generated: <b>${s.totalTokens}</b></span>
    <span>Succeeded: <b>${s.succeeded} / ${s.totalRequests}</b></span>
    <span>Wall time: <b>${s.wallTimeS.toFixed(2)}s</b></span>
  `;
}

function renderSweepSummary(sweepResults) {
  if (!sweepResults.length) return;
  const last = sweepResults[sweepResults.length - 1].summary;
  renderSingleSummary(last, last.concurrency);
  setEl("resultsTitle", "Concurrency Sweep Completed");

  const wrap = document.getElementById("sweepTableWrap");
  let html = `
    <table class="res-table">
      <thead>
        <tr>
          <th>Concurrency</th>
          <th>Throughput (tok/s)</th>
          <th>TTFT P50 (ms)</th>
          <th>TBT P50 (ms)</th>
          <th>Total Tok</th>
        </tr>
      </thead>
      <tbody>
  `;
  sweepResults.forEach(({ c, summary }) => {
    html += `
      <tr>
        <td>C = ${c}</td>
        <td><b>${summary.throughput.toFixed(1)}</b></td>
        <td>${summary.ttft.p50.toFixed(1)}</td>
        <td>${summary.tbt.p50.toFixed(1)}</td>
        <td>${summary.totalTokens}</td>
      </tr>
    `;
  });
  html += `</tbody></table>`;
  wrap.innerHTML = html;
}

function renderSweepChart(sweepResults) {
  const canvas = document.getElementById("sweepCanvas");
  if (!canvas || !sweepResults.length) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth || 320;
  const H = canvas.offsetHeight || 90;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const style = getComputedStyle(document.documentElement);
  const accent = style.getPropertyValue("--accent").trim() || "#00c9a7";
  const warn = style.getPropertyValue("--warn").trim() || "#f0a05a";

  const maxTps = Math.max(...sweepResults.map(r => r.summary.throughput), 1);
  const maxTtft = Math.max(...sweepResults.map(r => r.summary.ttft.p50), 1);
  const pad = 12;

  // Draw Throughput curve (Accent)
  ctx.beginPath();
  sweepResults.forEach(({ summary }, i) => {
    const x = pad + (i / Math.max(sweepResults.length - 1, 1)) * (W - pad * 2);
    const y = H - pad - (summary.throughput / maxTps) * (H - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw TTFT curve (Warn / Amber)
  ctx.beginPath();
  sweepResults.forEach(({ summary }, i) => {
    const x = pad + (i / Math.max(sweepResults.length - 1, 1)) * (W - pad * 2);
    const y = H - pad - (summary.ttft.p50 / maxTtft) * (H - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = warn;
  ctx.lineWidth = 1.6;
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Dots
  sweepResults.forEach(({ c, summary }, i) => {
    const x = pad + (i / Math.max(sweepResults.length - 1, 1)) * (W - pad * 2);
    const yTps = H - pad - (summary.throughput / maxTps) * (H - pad * 2);
    ctx.beginPath();
    ctx.arc(x, yTps, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = accent;
    ctx.fill();

    // label
    ctx.font = "9px " + (style.getPropertyValue("--mono") || "monospace");
    ctx.fillStyle = style.getPropertyValue("--fg-3") || "#888";
    ctx.fillText("C=" + c, x - 8, H - 2);
  });
}

// ── Interactive Playground (Try It) ─────────────────────────────────────
function setPromptPreset(text) {
  const area = document.getElementById("promptInput");
  area.value = text;
  area.focus();
}

async function startGen() {
  const prompt = document.getElementById("promptInput").value.trim();
  if (!prompt) return;
  const maxTok = parseInt(document.getElementById("maxTokens").value, 10);
  const temp = parseFloat(document.getElementById("tempSlider").value);

  // If server is not connected, attempt auto-connect
  if (!isConnected) {
    await connect();
  }

  document.getElementById("genErr").textContent = "";
  document.getElementById("genBtn").style.display = "none";
  document.getElementById("stopBtn").style.display = "";
  document.getElementById("outputBox").innerHTML = '<span class="cursor"></span>';
  document.getElementById("tryStats").innerHTML = "";

  playgroundAbortCtrl = new AbortController();
  const genStart = performance.now();
  let tokenCount = 0;
  let firstTokenTime = null;

  try {
    const resp = await fetch(getBaseUrl() + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: maxTok,
        temperature: temp,
        stream: true,
      }),
      signal: playgroundAbortCtrl.signal,
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error("HTTP " + resp.status + " — " + txt.slice(0, 100));
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let output = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        if (raw === "[DONE]") break;
        try {
          const obj = JSON.parse(raw);
          if (obj.object === "done") continue;
          if (obj.delta !== undefined) {
            if (firstTokenTime === null) firstTokenTime = performance.now();
            output += obj.delta;
            tokenCount++;
            const box = document.getElementById("outputBox");
            box.innerHTML = escHtml(output) + '<span class="cursor"></span>';
            box.scrollTop = box.scrollHeight;
          }
        } catch (_) {}
      }
    }

    const elapsed = (performance.now() - genStart) / 1000.0;
    const ttft = firstTokenTime ? (firstTokenTime - genStart).toFixed(0) : "—";
    const tps = elapsed > 0 ? (tokenCount / elapsed).toFixed(1) : "—";

    document.getElementById("tryStats").innerHTML = `
      <span class="try-stat">TTFT: <b>${ttft} ms</b></span>
      <span class="try-stat">Tokens: <b>${tokenCount}</b></span>
      <span class="try-stat">Throughput: <b>${tps} tok/s</b></span>
      <span class="try-stat">Latency: <b>${elapsed.toFixed(2)}s</b></span>
    `;

    document.getElementById("outputBox").innerHTML = escHtml(output) || '<span class="placeholder-text">Completed (empty generation)</span>';
  } catch (e) {
    if (e.name !== "AbortError") {
      document.getElementById("genErr").textContent = e.message;
      document.getElementById("outputBox").innerHTML =
        '<span style="color:var(--err)">Error communicating with server. Is the server running?</span>';
    }
  }

  document.getElementById("genBtn").style.display = "";
  document.getElementById("stopBtn").style.display = "none";
  playgroundAbortCtrl = null;
}

function stopGen() {
  if (playgroundAbortCtrl) playgroundAbortCtrl.abort();
}

// ── Window Resize & Init ────────────────────────────────────────────────
window.addEventListener("resize", () => {
  drawSparkline(tpsHistory);
});

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  renderPipeline();
  renderTests();
  initKvGrid(128);
  drawSparkline([]);

  // Auto-connect on page load
  try {
    await connect();
  } catch (_) {}
});
