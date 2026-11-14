# Real-World Benchmark Comparison: Naive HuggingFace vs. NanoInfer

| Engine | Concurrency | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | E2E P99 (ms) | KV Memory Waste (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive HuggingFace (model.generate)** | 1 | **75.6** | 32.1 | 118.1 | 401.2 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 1 | **73.7** | 54.0 | 137.6 | 439.1 | **25.2%** |
| **Naive HuggingFace (model.generate)** | 4 | **77.3** | 631.2 | 650.7 | 929.3 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 4 | **75.2** | 237.7 | 291.4 | 1357.3 | **25.2%** |
| **Naive HuggingFace (model.generate)** | 8 | **76.4** | 1438.9 | 1913.2 | 2194.5 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 8 | **75.0** | 469.6 | 470.2 | 2616.7 | **25.2%** |