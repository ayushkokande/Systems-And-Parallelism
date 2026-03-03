"""
Memory profiling script for the Basics Transformer model.
Times forward and backward passes across different model sizes.
"""

import argparse
import math
import timeit
from contextlib import nullcontext

import torch
import pandas as pd

from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

VOCAB_SIZE = 10_000
BATCH_SIZE = 4


def benchmark_model(cfg, context_length, warmup_steps, measure_steps, forward_only, device, mixed_precision=False, profile_memory=False, size_label="2.7B"):
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    optimizer = AdamW(model.parameters()) if not forward_only else None
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext())

    def step():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        with ctx:
            logits = model(input_ids)
        if not forward_only:
            logits.sum().backward()
            optimizer.step()
        torch.cuda.synchronize()

    for _ in range(warmup_steps):
        step()
    
    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(measure_steps):
        t0 = timeit.default_timer()
        step()
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024**2)

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))

    if profile_memory:
        snapshot_name = f"memory_{size_label}_ctx{context_length}_{'fwd' if forward_only else 'train'}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_name)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"  Snapshot saved: {snapshot_name}")

    return {
        "mode": "forward" if forward_only else "train",
        "mean_ms": mean,
        "peak_mb": peak_mb, 
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument("--size", nargs="+", choices=list(MODEL_CONFIGS) + ["all"], required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true", help="Run with BF16 mixed precision")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    if "all" in args.size:
        args.size = list(MODEL_CONFIGS)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    print(f"Device: {device}")
    print(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    rows = []
    for name in args.size:
        cfg = MODEL_CONFIGS[name]
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {name}")
        print(f"  context_length={args.context_length}, mixed_precision={args.mixed_precision}")
        print(f"{'=' * 60}")

        # --forward-only: run forward pass only. No flag: run full training step (fwd + backward + optimizer)
        forward_only = args.forward_only
        try:
            result = benchmark_model(
                cfg, args.context_length, args.warmup_steps,
                args.measure_steps, forward_only, device,
                args.mixed_precision, args.profile_memory, size_label=name
            )
            rows.append({
                "size_label": name,
                "mode": result["mode"],
                "mean_ms": result["mean_ms"],
                "peak_mb": result["peak_mb"],
                "context_length": args.context_length,
                "mixed_precision": args.mixed_precision,
            })
            print(f"  {result['mode']:30s}  mean={result['mean_ms']:8.2f} ms  peak={result['peak_mb']:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  {name} - {'forward' if forward_only else 'train'}: OOM (skipping)")
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n=== results ===")
    print(df.to_string(index=False))
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"saved to {args.output_csv}")

if __name__ == "__main__":
    main()