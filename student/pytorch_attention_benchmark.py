import torch
import math
import itertools

BATCH_SIZE = 8
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
NUM_WARMUP = 10
NUM_ITER = 100

def naive_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

def benchmark_config(d_model, seq_len, device):
    try:
        Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

        for _ in range(NUM_WARMUP):
            out = naive_attention(Q, K, V)
            out.sum().backward()
            Q.grad = K.grad = V.grad = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        
        start_fwd.record()
        for _ in range(NUM_ITER):
            output = naive_attention(Q, K, V)
        end_fwd.record()
        
        mem_before_bwd = torch.cuda.memory_allocated() / (1024**2)

        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)
        start_bwd.record()
        for _ in range(NUM_ITER):
            Q.grad = K.grad = V.grad = None
            naive_attention(Q, K, V).sum().backward()
        end_bwd.record()

        torch.cuda.synchronize()

        fwd_time = start_fwd.elapsed_time(end_fwd) / NUM_ITER
        bwd_time = start_bwd.elapsed_time(end_bwd) / NUM_ITER
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        return {
            "d_model": d_model, "seq": seq_len,
            "fwd_ms": fwd_time, "bwd_ms": bwd_time,
            "mem_before_bwd_mb": mem_before_bwd, "peak_mb": peak_mem
        }

    except torch.cuda.OutOfMemoryError:
        return "OOM"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: Running on CPU. Results will not reflect GPU bottlenecks.")

    results = []
    print(f"{'d_model':<10} | {'seq_len':<10} | {'fwd (ms)':<10} | {'bwd (ms)':<10} | {'Peak Mem (MB)':<15}")
    print("-" * 65)

    for d, s in itertools.product(D_MODEL_VALUES, SEQ_LEN_VALUES):
        res = benchmark_config(d, s, device)
        if res == "OOM":
            print(f"{d:<10} | {s:<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<15}")
            results.append({"d_model": d, "seq": s, "status": "OOM"})
        else:
            print(f"{d:<10} | {s:<10} | {res['fwd_ms']:<10.2f} | {res['bwd_ms']:<10.2f} | {res['peak_mb']:<15.2f}")
            results.append(res)

if __name__ == "__main__":
    main()