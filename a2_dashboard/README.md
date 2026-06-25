---
title: CS336 a2 GPU Performance Dashboard
emoji: ⚡
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
---

# CS336 a2 — GPU Performance Dashboard

Interactive dashboard over the benchmark results from CS336 assignment 2
(Systems & Parallelism). It visualizes **precomputed** GPU measurements — the
heavy benchmarking ran once on a GPU; this app only reads the saved CSVs and
draws charts, so it runs on a **free CPU host** with no GPU at view time.

## Tabs

| Tab | Shows |
|-----|-------|
| ⚡ Flash Attention | Custom **Triton FlashAttention** kernel vs naive PyTorch attention — fwd/bwd/e2e across dtype, head dim, sequence length (log-log). Naive OOMs at long sequences; flash keeps scaling. |
| 🎯 Mixed Precision | Full-model step time, **fp32 vs bf16** autocast, across context lengths and model sizes. |
| 🛠️ torch.compile | **Eager vs `torch.compile`** speedup across model sizes (forward / fwd+bwd / train). |
| 📈 Model Scaling | Latency vs context length per model size (small → 2.7B), with the **OOM frontier** marked. |

## Run locally

```bash
pip install -r requirements.txt
python app.py            # serves at http://localhost:7860
```

## Data

CSVs in `data/` are copied from `student/batch_results/` in the assignment repo:

- `flash_bench.csv` — Triton flash vs PyTorch attention
- `benchmark_fp32.csv`, `benchmark_bf16.csv` — model step-time benchmarks
- `compile_model.csv`, `compile_attn.csv` — eager vs compiled

To refresh: re-run the benchmark scripts in `student/` on a GPU and copy the new
CSVs into `data/`.

## Embed

```html
<iframe src="https://<user>-<space>.hf.space" width="100%" height="720"></iframe>
```
