import argparse
import math
import timeit
from contextlib import nullcontext

import torch
import pandas as pd

from a1_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

VOCAB_SIZE = 10_000
BATCH_SIZE = 4


def benchmark_model(cfg, context_length, warmup_steps, measure_steps, forward_only, device, mixed_precision=False):
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext())

    def step():
        with ctx:
            logits = model(input_ids)
        if not forward_only:
            logits.sum().backward()
        torch.cuda.synchronize()

    for _ in range(warmup_steps):
        step()
        model.zero_grad(set_to_none=True)

    times = []
    for _ in range(measure_steps):
        t0 = timeit.default_timer()
        step()
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)
        model.zero_grad(set_to_none=True)

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return {
        "mode": "forward" if forward_only else "forward+backward",
        "num_params_m": num_params / 1e6,
        "mean_ms": mean,
        "std_ms": std,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument("--size", nargs="+", choices=list(MODEL_CONFIGS) + ["all"], required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true", help="Run with BF16 mixed precision")
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    if "all" in args.size:
        args.size = list(MODEL_CONFIGS)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    print(f"Device: {device}")
    print(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    if args.mixed_precision:
        print("  Mixed precision: BF16 (torch.autocast)")
    else:
        print("  Mixed precision: Disabled")

    rows = []
    for name in args.size:
        cfg = MODEL_CONFIGS[name]
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {name} (d_model={cfg['d_model']}, layers={cfg['num_layers']}, "
              f"heads={cfg['num_heads']}, d_ff={cfg['d_ff']})")
        print(f"  context_length={args.context_length}, warmup={args.warmup_steps}, "
              f"measure={args.measure_steps}")
        print(f"{'=' * 60}")

        modes = [True] if args.forward_only else [True, False]
        for fwd_only in modes:
            result = benchmark_model(cfg, args.context_length, args.warmup_steps,
                                     args.measure_steps, fwd_only, device, args.mixed_precision)
            rows.append({
                "size_label": name,
                "mode": result["mode"],
                "num_params_m": result["num_params_m"],
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "mean_ms": result["mean_ms"],
                "std_ms": result["std_ms"],
                "context_length": args.context_length,
                "mixed_precision": args.mixed_precision,
            })
            print(f"  {result['mode']:20s}  mean={result['mean_ms']:8.2f} ms  "
                  f"std={result['std_ms']:6.2f} ms  params={result['num_params_m']:.1f}M")

    df = pd.DataFrame(rows)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
