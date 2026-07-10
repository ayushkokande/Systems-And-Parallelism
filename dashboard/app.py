"""
Systems & Parallelism — GPU performance dashboard.

Visualizes precomputed benchmark results from `data/`. All numbers were measured
once on a GPU; this app just reads the CSVs and draws charts, so it runs on a
free CPU host (HuggingFace Spaces) with no GPU required at view time.

Dual-mode: `python app.py` works locally and on Spaces (zero-arg launch).
"""

import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEMPLATE = "plotly_dark"
PALETTE = {"slow": "#ef553b", "fast": "#00cc96", "a": "#636efa", "b": "#ffa15a"}


def _csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def _num(series):
    """Coerce a column to float; 'FAILED' / 'N/A' / '' become NaN (OOM markers)."""
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------- #
# Data (loaded once at startup)
# --------------------------------------------------------------------------- #
FLASH = _csv("flash_bench.csv")
for c in ("fwd_ms", "bwd_ms", "e2e_ms"):
    FLASH[c] = _num(FLASH[c])

FP32 = _csv("benchmark_fp32.csv")
FP32["mean_ms"] = _num(FP32["mean_ms"])
BF16 = _csv("benchmark_bf16.csv")
BF16["mean_ms"] = _num(BF16["mean_ms"])

CMODEL = _csv("compile_model.csv")
CMODEL["mean_ms"] = _num(CMODEL["mean_ms"])
CATTN = _csv("compile_attn.csv")

SIZE_ORDER = ["small", "medium", "large", "xl", "2.7B"]
HEAD_DIMS = sorted(FLASH["d"].unique().tolist())


# --------------------------------------------------------------------------- #
# Tab 1 — Flash attention: Triton flash vs PyTorch naive
# --------------------------------------------------------------------------- #
def flash_fig(dtype, d, metric):
    sub = FLASH[(FLASH["dtype"] == dtype) & (FLASH["d"] == int(d))].sort_values("seq_len")
    fig = go.Figure()
    for impl, color, label in (
        ("pytorch", PALETTE["slow"], "PyTorch naive"),
        ("triton_flash", PALETTE["fast"], "Triton FlashAttention"),
    ):
        s = sub[sub["impl"] == impl].dropna(subset=[metric])
        fig.add_trace(go.Scatter(
            x=s["seq_len"], y=s[metric], mode="lines+markers", name=label,
            line=dict(color=color, width=3), marker=dict(size=7),
        ))

    # Speedup callout: where both implementations have a value, max pytorch/flash.
    pt = sub[sub["impl"] == "pytorch"].set_index("seq_len")[metric]
    fl = sub[sub["impl"] == "triton_flash"].set_index("seq_len")[metric]
    ratio = (pt / fl).dropna()
    note = f"peak speedup ≈ {ratio.max():.1f}× (seq {int(ratio.idxmax())})" if len(ratio) else ""

    # Longest seq each impl reaches (naive OOMs earlier).
    pt_max = sub[sub["impl"] == "pytorch"].dropna(subset=[metric])["seq_len"].max()
    fl_max = sub[sub["impl"] == "triton_flash"].dropna(subset=[metric])["seq_len"].max()
    if pd.notna(pt_max) and pd.notna(fl_max) and fl_max > pt_max:
        note += f"  •  naive OOM past seq {int(pt_max)}, flash reaches {int(fl_max)}"

    fig.update_layout(
        template=TEMPLATE, title=f"{metric} — {dtype}, head dim {d}<br><sup>{note}</sup>",
        xaxis_title="sequence length", yaxis_title="time (ms)",
        xaxis_type="log", yaxis_type="log", legend=dict(x=0.02, y=0.98),
        margin=dict(t=70, l=60, r=20, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Tab 2 — Mixed precision: fp32 vs bf16 for one model size
# --------------------------------------------------------------------------- #
def mp_fig(size, mode):
    a = FP32[(FP32["size_label"] == size) & (FP32["mode"] == mode)][["context_length", "mean_ms"]]
    b = BF16[(BF16["size_label"] == size) & (BF16["mode"] == mode)][["context_length", "mean_ms"]]
    m = a.merge(b, on="context_length", how="outer", suffixes=("_fp32", "_bf16")).sort_values("context_length")

    fig = go.Figure()
    fig.add_bar(x=m["context_length"], y=m["mean_ms_fp32"], name="fp32", marker_color=PALETTE["a"])
    fig.add_bar(x=m["context_length"], y=m["mean_ms_bf16"], name="bf16", marker_color=PALETTE["b"])

    # Annotate context lengths where fp32 OOMed but we have a row (mean is NaN).
    oom = m[m["mean_ms_fp32"].isna() & m["context_length"].isin(FP32[FP32["size_label"] == size]["context_length"])]
    for ctx in oom["context_length"]:
        fig.add_annotation(x=ctx, y=0, text="fp32 OOM", showarrow=False, yshift=14,
                           font=dict(color=PALETTE["slow"], size=11))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title=f"fp32 vs bf16 — {size}, {mode}",
        xaxis_title="context length", yaxis_title="mean step time (ms)",
        xaxis=dict(type="category"), margin=dict(t=60, l=60, r=20, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Tab 3 — torch.compile: eager vs compiled (full model)
# --------------------------------------------------------------------------- #
def compile_fig(group):
    eager_mode, comp_mode = group, f"{group}+compiled"
    rows = []
    for size in SIZE_ORDER:
        e = CMODEL[(CMODEL["size"] == size) & (CMODEL["mode"] == eager_mode)]["mean_ms"]
        c = CMODEL[(CMODEL["size"] == size) & (CMODEL["mode"] == comp_mode)]["mean_ms"]
        if len(e) and len(c):
            rows.append((size, float(e.iloc[0]), float(c.iloc[0])))
    if not rows:
        return go.Figure().update_layout(template=TEMPLATE, title="no data for this mode")

    sizes = [r[0] for r in rows]
    eager = [r[1] for r in rows]
    comp = [r[2] for r in rows]
    speed = [f"{e / c:.1f}×" for e, c in zip(eager, comp)]

    fig = go.Figure()
    fig.add_bar(x=sizes, y=eager, name="eager", marker_color=PALETTE["slow"])
    fig.add_bar(x=sizes, y=comp, name="compiled", marker_color=PALETTE["fast"], text=speed, textposition="outside")
    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title=f"torch.compile speedup — {group} (ctx 128)<br><sup>label = eager/compiled</sup>",
        xaxis_title="model size", yaxis_title="mean time (ms)",
        margin=dict(t=70, l=60, r=20, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# Tab 4 — Model scaling + OOM frontier
# --------------------------------------------------------------------------- #
def scaling_fig(precision, mode):
    df = FP32 if precision == "fp32" else BF16
    sub = df[df["mode"] == mode]
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, size in enumerate(SIZE_ORDER):
        s = sub[sub["size_label"] == size].sort_values("context_length")
        ok = s.dropna(subset=["mean_ms"])
        if len(ok):
            fig.add_trace(go.Scatter(
                x=ok["context_length"], y=ok["mean_ms"], mode="lines+markers",
                name=size, line=dict(color=colors[i % len(colors)], width=2.5),
            ))
        # Mark first OOM context for this size.
        oom = s[s["mean_ms"].isna()]["context_length"]
        if len(oom):
            fig.add_annotation(x=int(oom.min()), y=0, text=f"{size} OOM", showarrow=False,
                               yshift=10 + 14 * i, font=dict(color=colors[i % len(colors)], size=10))

    fig.update_layout(
        template=TEMPLATE, title=f"Latency vs context length — {precision}, {mode}",
        xaxis_title="context length", yaxis_title="mean step time (ms)",
        xaxis=dict(type="category"), yaxis_type="log",
        legend=dict(title="model size"), margin=dict(t=60, l=60, r=20, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
INTRO = """
# GPU Performance Dashboard
**Systems & Parallelism.** Interactive view of real benchmark results from a
from-scratch Transformer: a custom **Triton FlashAttention** kernel, **mixed
precision**, **`torch.compile`**, and **model scaling** to the OOM frontier.

All numbers were measured once on a GPU. This dashboard reads the saved CSVs —
no GPU needed to view it.
"""

with gr.Blocks(title="GPU Performance Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("⚡ Flash Attention"):
        gr.Markdown("Custom Triton FlashAttention vs naive PyTorch attention. Log-log axes.")
        with gr.Row():
            f_dtype = gr.Radio(["float32", "bfloat16"], value="bfloat16", label="dtype")
            f_d = gr.Dropdown([str(d) for d in HEAD_DIMS], value=str(HEAD_DIMS[-1]), label="head dim")
            f_metric = gr.Radio(["fwd_ms", "bwd_ms", "e2e_ms"], value="fwd_ms", label="metric")
        f_plot = gr.Plot()
        for c in (f_dtype, f_d, f_metric):
            c.change(flash_fig, [f_dtype, f_d, f_metric], f_plot)
        demo.load(flash_fig, [f_dtype, f_d, f_metric], f_plot)

    with gr.Tab("🎯 Mixed Precision"):
        gr.Markdown("Full-model step time: fp32 vs bf16 autocast across context lengths.")
        with gr.Row():
            m_size = gr.Dropdown(SIZE_ORDER, value="medium", label="model size")
            m_mode = gr.Radio(["forward", "forward+backward"], value="forward+backward", label="mode")
        m_plot = gr.Plot()
        for c in (m_size, m_mode):
            c.change(mp_fig, [m_size, m_mode], m_plot)
        demo.load(mp_fig, [m_size, m_mode], m_plot)

    with gr.Tab("🛠️ torch.compile"):
        gr.Markdown("Eager vs `torch.compile`d, full model, across sizes (context 128).")
        c_group = gr.Radio(["forward", "forward+backward", "train"], value="forward", label="pass")
        c_plot = gr.Plot()
        c_group.change(compile_fig, c_group, c_plot)
        demo.load(compile_fig, c_group, c_plot)

    with gr.Tab("📈 Model Scaling"):
        gr.Markdown("Latency vs context length per model size; annotations mark the OOM frontier.")
        with gr.Row():
            s_prec = gr.Radio(["fp32", "bf16"], value="fp32", label="precision")
            s_mode = gr.Radio(["forward", "forward+backward"], value="forward", label="mode")
        s_plot = gr.Plot()
        for c in (s_prec, s_mode):
            c.change(scaling_fig, [s_prec, s_mode], s_plot)
        demo.load(scaling_fig, [s_prec, s_mode], s_plot)

    gr.Markdown(
        "<sub>Source: `systems/batch_results/`. Charts: Plotly. "
        "From the Systems & Parallelism repository.</sub>"
    )


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        # HuggingFace Spaces sets server name/port via env; let Gradio use them.
        demo.launch()
    else:
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
