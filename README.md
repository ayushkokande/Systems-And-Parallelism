# Systems & Parallelism — GPU Performance Engineering for Transformer LMs

Low-level performance work on a from-scratch Transformer language model:
a custom **FlashAttention-2** implementation (pure PyTorch and **Triton
kernels**), systematic **benchmarking**, **Nsight Systems and CUDA memory
profiling**, **mixed-precision (bf16)** training, and **`torch.compile`**
comparisons — measured across model sizes (small → 2.7B) and context lengths
up to the OOM frontier on real GPUs.

**Live results:** interactive dashboard at
[ayushkokande-systems-and-parallelism.hf.space](https://ayushkokande-systems-and-parallelism.hf.space)
(precomputed measurements, no GPU needed to view).

## What's implemented

- [`systems/flash_attention.py`](./systems/flash_attention.py) — FlashAttention-2
  forward and backward as a `torch.autograd.Function`, in pure PyTorch (tiled,
  online softmax, logsumexp recomputation).
- [`systems/flash_attention_triton.py`](./systems/flash_attention_triton.py) —
  the same algorithm with the forward pass as a handwritten **Triton kernel**.
- [`systems/benchmark.py`](./systems/benchmark.py) — end-to-end model step-time
  benchmarking harness (warmup control, fwd / fwd+bwd, CSV output).
- [`systems/memory_profiling.py`](./systems/memory_profiling.py) — CUDA memory
  snapshots across context lengths and precisions.
- [`systems/nsys_profile.py`](./systems/nsys_profile.py) — NVTX-annotated runs
  for Nsight Systems timeline analysis.
- [`systems/torch_compile_benchmark.py`](./systems/torch_compile_benchmark.py) —
  eager vs compiled, attention-only and full model.
- [`systems/mixed_precision_accumulation.py`](./systems/mixed_precision_accumulation.py) —
  fp16/bf16 accumulation behavior experiments.
- [`dashboard/`](./dashboard) — Gradio app visualizing all saved measurements
  (deployed on HuggingFace Spaces).

Measurements were collected on SLURM-managed NVIDIA GPUs; the batch scripts are
in [`systems/batch_scripts/`](./systems/batch_scripts) and the resulting CSVs in
`systems/batch_results/`.

## Layout

- [`./lm-basics`](./lm-basics) — the `lm_basics` package: reference Transformer
  LM (model, AdamW, data utilities) that the benchmarks target.
- [`./systems`](./systems) — the performance work listed above.
- [`./examples`](./examples) — small standalone benchmarking/profiling examples
  (kernel fusion, CUDA GELU, PyTorch profiler).
- [`./tests`](./tests) — correctness tests for the FlashAttention
  implementations against a naive attention reference.

## Setup

Dependencies are managed with `uv`; `uv run` builds both packages and installs
everything automatically:

```sh
$ uv run pytest tests/          # FlashAttention correctness (needs a GPU for the Triton tests)
$ uv run python -m systems.benchmark --size small --context-length 256
```

## Credits

The test scaffolding and reference LM are adapted from Stanford's
[CS336](https://github.com/stanford-cs336/) course materials.
