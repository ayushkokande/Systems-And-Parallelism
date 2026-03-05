import itertools
import math
import argparse

import torch
import triton
import pandas as pd

from student.flash_attention_triton import TritonFlashAttentionFunction


BATCH_SIZE = 1
SEQ_LENS = [2**i for i in range(7, 17)]
EMBED_DIMS = [2**i for i in range(4, 8)]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    S = torch.matmul(q, k.transpose(-2, -1)) * scale
    q_idx = torch.arange(S.shape[-2], device=S.device).unsqueeze(1)
    k_idx = torch.arange(S.shape[-1], device=S.device).unsqueeze(0)
    S = torch.where(q_idx >= k_idx, S, -1e6)
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, v)


def bench_config(seq_len, d, dtype, device):
    q = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)

    results = {}

    for label, attn_fn in [("pytorch", naive_attention), ("triton_flash", TritonFlashAttentionFunction.apply)]:
        is_flash = label == "triton_flash"

        def fwd():
            if is_flash:
                return attn_fn(q, k, v, True)
            return attn_fn(q, k, v)

        def fwd_bwd():
            q.grad = k.grad = v.grad = None
            o = fwd()
            o.backward(do)

        try:
            fwd_ms = triton.testing.do_bench(fwd, warmup=100, rep=200)
        except torch.cuda.OutOfMemoryError:
            results[label] = {"fwd_ms": None, "bwd_ms": None, "e2e_ms": None}
            torch.cuda.empty_cache()
            continue

        try:
            e2e_ms = triton.testing.do_bench(fwd_bwd, warmup=100, rep=200)
            bwd_ms = e2e_ms - fwd_ms
        except torch.cuda.OutOfMemoryError:
            bwd_ms = None
            e2e_ms = None
            torch.cuda.empty_cache()

        results[label] = {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "e2e_ms": e2e_ms}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    rows = []
    for dtype_name, dtype in DTYPES.items():
        for d, seq in itertools.product(EMBED_DIMS, SEQ_LENS):
            results = bench_config(seq, d, dtype, device)
            for impl, times in results.items():
                rows.append({
                    "dtype": dtype_name, "d": d, "seq_len": seq, "impl": impl,
                    "fwd_ms": times["fwd_ms"], "bwd_ms": times["bwd_ms"],
                    "e2e_ms": times["e2e_ms"],
                })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
