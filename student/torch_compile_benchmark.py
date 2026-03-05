import argparse
import itertools
import math
import sys
import timeit

import torch
import pandas as pd

from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW

ATTN_BATCH_SIZE = 8
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
ATTN_WARMUP = 10
ATTN_ITER = 100

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
MODEL_VOCAB_SIZE = 10_000
MODEL_BATCH_SIZE = 4

def naive_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


compiled_attention = torch.compile(naive_attention)


def bench_attention_config(d_model, seq_len, device, attn_fn, label):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        Q = torch.randn(ATTN_BATCH_SIZE, seq_len, d_model, device=device,
                         dtype=torch.float32, requires_grad=True)
        K = torch.randn_like(Q, requires_grad=True)
        V = torch.randn_like(Q, requires_grad=True)

        for _ in range(ATTN_WARMUP):
            attn_fn(Q, K, V).sum().backward()
            Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_fwd.record()
        for _ in range(ATTN_ITER):
            attn_fn(Q, K, V)
        end_fwd.record()
        torch.cuda.synchronize()
        fwd_ms = start_fwd.elapsed_time(end_fwd) / ATTN_ITER

        _ = attn_fn(Q, K, V).sum()
        mem_before_bwd_mb = torch.cuda.memory_allocated(device) / (1024**2)

        def fwd_bwd():
            Q.grad = K.grad = V.grad = None
            attn_fn(Q, K, V).sum().backward()

        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)
        start_bwd.record()
        for _ in range(ATTN_ITER):
            fwd_bwd()
        end_bwd.record()
        torch.cuda.synchronize()
        bwd_ms = start_bwd.elapsed_time(end_bwd) / ATTN_ITER - fwd_ms
        if bwd_ms < 0:
            bwd_ms = start_bwd.elapsed_time(end_bwd) / ATTN_ITER

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        return {
            "mode": label, "d_model": d_model, "seq": seq_len,
            "fwd_ms": fwd_ms, "bwd_ms": bwd_ms,
            "mem_before_bwd_mb": mem_before_bwd_mb, "peak_mb": peak_mb,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"


def run_part_a(device):
    print("=" * 80)
    print("Part (a): Attention – eager vs torch.compile")
    print("=" * 80)
    header = f"{'mode':<10} | {'d_model':<8} | {'seq':<8} | {'fwd(ms)':<10} | {'bwd(ms)':<10} | {'mem_bwd':<10} | {'peak':<10}"
    print(header)
    print("-" * len(header))

    rows = []
    for d, s in itertools.product(D_MODEL_VALUES, SEQ_LEN_VALUES):
        for label, fn in [("eager", naive_attention), ("compiled", compiled_attention)]:
            res = bench_attention_config(d, s, device, fn, label)
            if res == "OOM":
                print(f"{label:<10} | {d:<8} | {s:<8} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10}")
                rows.append({"mode": label, "d_model": d, "seq": s,
                             "fwd_ms": None, "bwd_ms": None,
                             "mem_before_bwd_mb": None, "peak_mb": None})
            else:
                print(f"{label:<10} | {d:<8} | {s:<8} | {res['fwd_ms']:<10.3f} | {res['bwd_ms']:<10.3f} | {res['mem_before_bwd_mb']:<10.2f} | {res['peak_mb']:<10.2f}")
                rows.append(res)

    return pd.DataFrame(rows)



def bench_model(cfg, context_length, warmup_steps, measure_steps,
                mode, device, use_compile=False):
    model = BasicsTransformerLM(
        vocab_size=MODEL_VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    if use_compile:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters()) if mode == "train" else None
    num_params = sum(p.numel() for p in model.parameters())
    input_ids = torch.randint(0, MODEL_VOCAB_SIZE, (MODEL_BATCH_SIZE, context_length), device=device)

    do_backward = mode in ("forward+backward", "train")

    def step():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        if do_backward:
            logits.sum().backward()
        if optimizer is not None:
            optimizer.step()
        torch.cuda.synchronize()

    for _ in range(warmup_steps):
        step()
        if optimizer is None:
            model.zero_grad(set_to_none=True)

    times = []
    for _ in range(measure_steps):
        t0 = timeit.default_timer()
        step()
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)
        if optimizer is None:
            model.zero_grad(set_to_none=True)

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    compile_label = "+compiled" if use_compile else ""
    return {
        "mode": mode + compile_label,
        "num_params_m": num_params / 1e6,
        "mean_ms": mean,
        "std_ms": std,
    }


def run_part_b(sizes, context_length, warmup_steps, measure_steps, device):
    print("\n" + "=" * 80)
    print("Part (b): Full Transformer – vanilla vs torch.compile")
    print("=" * 80)

    rows = []
    bench_modes = ["forward", "forward+backward", "train"]
    for name in sizes:
        cfg = MODEL_CONFIGS[name]
        print(f"\n--- {name} (d_model={cfg['d_model']}, layers={cfg['num_layers']}) ---")

        for mode in bench_modes:
            for use_compile in [False, True]:
                try:
                    result = bench_model(cfg, context_length, warmup_steps,
                                         measure_steps, mode, device,
                                         use_compile=use_compile)
                    rows.append({
                        "size": name,
                        "mode": result["mode"],
                        "num_params_m": result["num_params_m"],
                        "mean_ms": result["mean_ms"],
                        "std_ms": result["std_ms"],
                        "context_length": context_length,
                    })
                    print(f"  {result['mode']:30s}  mean={result['mean_ms']:8.2f} ms  std={result['std_ms']:6.2f} ms")
                except torch.cuda.OutOfMemoryError:
                    compile_label = "+compiled" if use_compile else ""
                    print(f"  {mode + compile_label:30s}  OOM")
                    torch.cuda.empty_cache()

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="torch.compile benchmarks (attention + full model)")
    parser.add_argument("--part", choices=["a", "b", "both"], default="both",
                        help="Which part to run: a (attention), b (full model), or both")
    parser.add_argument("--size", nargs="+", choices=list(MODEL_CONFIGS) + ["all"], default=["all"],
                        help="Model sizes for part (b)")
    parser.add_argument("--context-length", type=int, default=128, help="Context length for part (b)")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--output-csv-a", type=str, default=None, help="CSV output for part (a)")
    parser.add_argument("--output-csv-b", type=str, default=None, help="CSV output for part (b)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required.", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)\n")

    if "all" in args.size:
        args.size = list(MODEL_CONFIGS)

    if args.part in ("a", "both"):
        df_a = run_part_a(device)
        print("\n=== Part (a) Results ===")
        print(df_a.to_string(index=False))
        if args.output_csv_a:
            df_a.to_csv(args.output_csv_a, index=False)
            print(f"Saved to {args.output_csv_a}")

    if args.part in ("b", "both"):
        df_b = run_part_b(args.size, args.context_length, args.warmup_steps,
                          args.measure_steps, device)
        print("\n=== Part (b) Results ===")
        print(df_b.to_string(index=False))
        if args.output_csv_b:
            df_b.to_csv(args.output_csv_b, index=False)
            print(f"Saved to {args.output_csv_b}")


if __name__ == "__main__":
    main()
