"""
profiling script for transformer model
"""

import argparse
import math
import timeit

import torch
import pandas as pd

import a1_basics.model as basics_model
from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW

try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except:
    HAS_NVTX = False


MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4


def patch_with_nvtx():
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        nvtx.range_push("scaled dot product attention")
        
        d_k = K.shape[-1]
        
        nvtx.range_push("computing attention scores")
        attn_scores = basics_model.einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)
        if mask is not None:
            attn_scores = torch.where(mask, attn_scores, float("-inf"))
        nvtx.range_pop()

        nvtx.range_push("computing softmax")
        attn_weights = basics_model.softmax(attn_scores, dim=-1)
        nvtx.range_pop()

        nvtx.range_push("final matmul")
        out = basics_model.einsum(
            attn_weights, V, "... query key, ... key d_v -> ... query d_v"
        )
        nvtx.range_pop()

        nvtx.range_pop()  
        return out

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def run_benchmark(cfg, context_length, warmup_steps, measure_steps, forward_only, do_optimizer, device, use_nvtx):
    
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    if do_optimizer:
        optimizer = AdamW(model.parameters())
    
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)

    if forward_only:
        mode_str = "forward"
    elif do_optimizer:
        mode_str = "forward+backward+optimizer"
    else:
        mode_str = "forward+backward"

    def do_step():
        if use_nvtx:
            nvtx.range_push("forward")
        logits = model(input_ids)
        if use_nvtx:
            nvtx.range_pop()

        if not forward_only:
            if use_nvtx:
                nvtx.range_push("backward")
            logits.sum().backward()
            if use_nvtx:
                nvtx.range_pop()

            if do_optimizer:
                if use_nvtx:
                    nvtx.range_push("optimizer step")
                optimizer.step()
                if use_nvtx:
                    nvtx.range_pop()

        torch.cuda.synchronize()

    if use_nvtx:
        nvtx.range_push("warmup")
    for i in range(warmup_steps):
        do_step()
        model.zero_grad(set_to_none=True)
    if use_nvtx:
        nvtx.range_pop()

    times = []
    if use_nvtx:
        nvtx.range_push("measure")
    for i in range(measure_steps):
        t_start = timeit.default_timer()
        do_step()
        t_end = timeit.default_timer()
        times.append((t_end - t_start) * 1000) 
        model.zero_grad(set_to_none=True)
    if use_nvtx:
        nvtx.range_pop()

    mean_t = sum(times) / len(times)
    std_t = math.sqrt(sum((t - mean_t)**2 for t in times) / len(times))

    return mode_str, mean_t, std_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs="+", choices=list(MODEL_CONFIGS) + ["all"], required=True)
    parser.add_argument("--context-length", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")# store_true means the argument is a boolean and will be set to True if the argument is present
    parser.add_argument("--optimizer-step", action="store_true")# store_true means the argument is a boolean and will be set to True if the argument is present
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    if "all" in args.size:
        args.size = list(MODEL_CONFIGS.keys())

    assert torch.cuda.is_available(), "need cuda!"
    device = torch.device("cuda")

    use_nvtx = args.nvtx and HAS_NVTX
    if use_nvtx:
        patch_with_nvtx()

    rows = []

    for name in args.size:
        cfg = MODEL_CONFIGS[name]
        for ctx_len in args.context_length:
            print(f"\nrunning: {name}, context_length={ctx_len}")

            try:
                mode, mean_t, std_t = run_benchmark(
                    cfg, ctx_len, args.warmup_steps, args.measure_steps,
                    forward_only=True, do_optimizer=False,
                    device=device, use_nvtx=use_nvtx
                )
                print(f"  {mode}: {mean_t:.2f}ms (std={std_t:.2f})")
                rows.append({"size": name, "ctx_len": ctx_len, "mode": mode, "mean_ms": mean_t, "status": "ok"})
            except torch.cuda.OutOfMemoryError:
                print(f"  forward: OOM")
                rows.append({"size": name, "ctx_len": ctx_len, "mode": "forward", "mean_ms": float("nan"), "status": "OOM"})
                torch.cuda.empty_cache()
                continue  

            if not args.forward_only:
                bwd_mode = "forward+backward+optimizer" if args.optimizer_step else "forward+backward"
                try:
                    mode, mean_t, std_t = run_benchmark(
                        cfg, ctx_len, args.warmup_steps, args.measure_steps,
                        forward_only=False, do_optimizer=args.optimizer_step,
                        device=device, use_nvtx=use_nvtx
                    )
                    print(f"  {mode}: {mean_t:.2f}ms (std={std_t:.2f})")
                    rows.append({"size": name, "ctx_len": ctx_len, "mode": mode, "mean_ms": mean_t, "status": "ok"})
                except torch.cuda.OutOfMemoryError:
                    print(f"  {bwd_mode}: OOM")
                    rows.append({"size": name, "ctx_len": ctx_len, "mode": bwd_mode, "mean_ms": float("nan"), "status": "OOM"})
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n=== results ===")
    print(df.to_string(index=False))

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"saved to {args.output_csv}")


if __name__ == "__main__":
    main()
