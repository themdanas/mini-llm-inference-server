# Real-World Benchmark Comparison: Naive HuggingFace vs. NanoInfer

| Engine | Concurrency | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | E2E P99 (ms) | KV Memory Waste (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive HuggingFace (model.generate)** | 1 | **85.8** | 27.5 | 76.0 | 328.2 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 1 | **81.4** | 50.1 | 59.3 | 319.8 | **25.2%** |
| **Naive HuggingFace (model.generate)** | 4 | **84.0** | 588.7 | 600.3 | 853.1 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 4 | **90.3** | 187.5 | 251.4 | 1203.8 | **25.2%** |
| **Naive HuggingFace (model.generate)** | 8 | **85.2** | 1288.7 | 1716.8 | 1968.1 | **98.3%** |
| **NanoInfer (Paged KV + Cont. Batch)** | 8 | **117.6** | 342.0 | 342.3 | 1422.5 | **25.2%** |