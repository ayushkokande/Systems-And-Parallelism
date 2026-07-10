# Portfolio Deploy Playbook

Turn any ML project into a hosted, embeddable demo. Distilled from shipping a
from-scratch Transformer LM as a live HuggingFace Space embedded in a portfolio site.

---

## The 5-step recipe (any ML project → portfolio demo)

### Step 1 — find or build the inference entrypoint
One function: `input → load model → predict → output`. Most projects already
have it (a CLI, a test, a `decode.py`). If not, write it. This is the thing the
UI wraps. Keep it import-safe (no side effects at module load).

### Step 2 — wrap in Gradio (`app.py`)
```python
import gradio as gr

def predict(x):
    return model(x)          # your inference fn from step 1

demo = gr.Interface(fn=predict, inputs=..., outputs=...)
demo.launch()
```
- Match inputs/outputs to the model: `Textbox`, `Image`, `Audio`, `Slider`.
- `gr.Interface` for quick; `gr.Blocks` for custom layout.
- Generator + `yield` = live token streaming.

### Step 3 — make it Spaces-ready
- Entry file **must** be `app.py`.
- `requirements.txt` = runtime deps only (no `wandb`, no CUDA-train stuff).
- `README.md` with HF YAML frontmatter (`sdk: gradio`, `app_file: app.py`).
- Zero-arg launch: load weights from a fixed `model/` dir (Spaces runs
  `python app.py` with no CLI args). Detect `SPACE_ID` env var to branch.
- Commit model weights via Git LFS.

### Step 4 — deploy free
HuggingFace Spaces, CPU-basic tier ($0). Push the repo. Live in ~2–3 min at a
permanent `https://<user>-<space>.hf.space`.

### Step 5 — embed
```html
<iframe src="https://<user>-<space>.hf.space" width="100%" height="640"></iframe>
```

---

## Gotchas

| Gotcha | Fix |
|--------|-----|
| Training compute ≠ deploy compute | Train on GPU (RunPod), host on free CPU (keep model small) |
| Checkpoint carries optimizer state (~3× size) | Strip to weights-only before upload |
| Package `__init__` assumes pip-install (metadata/version lookup) | Guard the lookup so it imports as a plain folder |
| Spaces runs `python app.py` with no args | Default paths via env var + `model/` dir; detect `SPACE_ID` |
| Live GPU inference too heavy for free CPU | Quantize/distill down, or precompute results and visualize them |

---

## Hosting decision table

| Model size | Where | Cost |
|------------|-------|------|
| Small (<100M, CPU-runnable) | HF Spaces free CPU | $0 |
| Medium (needs GPU, occasional) | HF Spaces ZeroGPU / paid T4 | low |
| Heavy / always-on GPU | Dedicated pod, Modal, Replicate | $$ |

For a portfolio, aim for "small enough to run on free CPU." If inference needs a
GPU, either shrink the model, or use pay-per-call infra (Replicate/Modal) that
scales to zero.

---

## When the project isn't generative (systems/benchmark work)

Not every project is "type prompt → get output." For benchmarking, profiling, or
systems work (like this repo), the deliverable is **measurements**, not a live
model. The portfolio move:

1. Run the expensive benchmarks **once** on a GPU; save results (CSV/pickle).
2. Build a UI that **visualizes the saved results** — interactive charts, not
   live compute. Host on free CPU (no GPU needed at view time).

Same "train on GPU, host on free CPU" split — just precomputed data instead of
precomputed weights.
