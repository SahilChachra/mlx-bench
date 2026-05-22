"""
Generate HuggingFace model cards (README.md) for each quantized model.
Reads from outputs/<model>/summary.json and writes README.md into models/<model>/.

Usage:
  python scripts/generate_model_cards.py
"""

import json
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR  = Path(__file__).parent.parent / "models"

AUTHOR      = "sahilchachra"
BASE_MODEL  = "ibm-granite/granite-4.1-8b"
BASE_LABEL  = "granite-4.1-8b-fp16"

# All variants we publish — used for cross-links
ALL_VARIANTS = [
    ("4bit",      "Affine int4 (group 64)"),
    ("5bit",      "Affine int5 (group 64)"),
    ("6bit",      "Affine int6 (group 64)"),
    ("8bit",      "Affine int8 (group 64)"),
    ("mixed4_6",  "Mixed 4+6 bit"),
    ("mxfp4",     "Block float MX FP4"),
    ("mxfp8",     "Block float MX FP8"),
]

QUANT_DESCRIPTIONS = {
    "4bit": {
        "method": "Affine integer quantization",
        "bits": "4-bit (4.5 bits/weight avg)",
        "group_size": 64,
        "description": (
            "Standard affine (integer) quantization at 4-bit with group size 64. "
            "Largest compression ratio of the uniform variants. "
            "~3.9× smaller than FP16 with moderate quality tradeoff."
        ),
    },
    "5bit": {
        "method": "Affine integer quantization",
        "bits": "5-bit (~5.5 bits/weight avg)",
        "group_size": 64,
        "description": (
            "Affine quantization at 5-bit with group size 64. "
            "Good middle ground between 4-bit compression and 6-bit quality."
        ),
    },
    "6bit": {
        "method": "Affine integer quantization",
        "bits": "6-bit (~6.5 bits/weight avg)",
        "group_size": 64,
        "description": (
            "Affine quantization at 6-bit with group size 64. "
            "Best quality among the smaller uniform variants, recommended for general use."
        ),
    },
    "8bit": {
        "method": "Affine integer quantization",
        "bits": "8-bit (~8.5 bits/weight avg)",
        "group_size": 64,
        "description": (
            "Affine quantization at 8-bit with group size 64. "
            "Closest to FP16 quality. Recommended when memory allows and quality is the priority."
        ),
    },
    "mixed4_6": {
        "method": "Mixed-bit quantization (mlx-lm predicate: mixed_4_6)",
        "bits": "~4.77 bits/weight avg",
        "group_size": 64,
        "description": (
            "Mixed-bit quantization where sensitive layers (embeddings, first/last transformer layers) "
            "are kept at 6-bit while the remaining layers use 4-bit. "
            "Achieves better quality than uniform 4-bit at nearly the same disk size (~4.8 GB vs ~4.5 GB)."
        ),
    },
    "mxfp4": {
        "method": "Block floating-point MX FP4 (microscaling)",
        "bits": "~4 bits/weight",
        "group_size": 32,
        "description": (
            "Microscaling (MX) block floating-point quantization at FP4 precision. "
            "Uses a shared floating-point exponent per block of 32 weights instead of integer affine scaling. "
            "Different numerical properties vs affine int4 — may suit different workloads."
        ),
    },
    "mxfp8": {
        "method": "Block floating-point MX FP8 (microscaling)",
        "bits": "~8 bits/weight",
        "group_size": 32,
        "description": (
            "Microscaling (MX) block floating-point quantization at FP8 precision. "
            "Uses a shared floating-point exponent per block of 32 weights. "
            "Compared to affine int8: same bit-width, different numerical format."
        ),
    },
}


def load_summary(label):
    p = OUTPUTS_DIR / f"granite-4.1-8b-{label}" / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_fp16_summary():
    return load_summary("fp16")


def fmt(v, suffix="", missing="N/A"):
    return f"{v}{suffix}" if v is not None else missing


def disk_size_mb(label):
    p = MODELS_DIR / f"granite-4.1-8b-{label}"
    if not p.exists():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**2)


def generate_card(label):
    model_name = f"granite-4.1-8b-{label}"
    repo_name  = f"{AUTHOR}/{model_name}-mlx"
    qinfo      = QUANT_DESCRIPTIONS[label]
    s          = load_summary(label)
    fp16       = load_fp16_summary()
    disk       = disk_size_mb(label)

    b  = s.get("benchmarks", {}) if s else {}
    p  = s.get("perf", {}) if s else {}
    fp = fp16.get("perf", {}) if fp16 else {}
    fb = fp16.get("benchmarks", {}) if fp16 else {}

    lines = []
    a = lines.append

    a(f"---")
    a(f"language: en")
    a(f"license: apache-2.0")
    a(f"base_model: {BASE_MODEL}")
    a(f"tags:")
    a(f"  - mlx")
    a(f"  - quantized")
    a(f"  - granite")
    a(f"  - apple-silicon")
    a(f"---")
    a(f"")
    a(f"# {model_name}-mlx")
    a(f"")
    a(f"Quantized version of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for Apple Silicon using [MLX](https://github.com/ml-explore/mlx).")
    a(f"")
    a(f"**Quantization**: {qinfo['method']}  ")
    a(f"**Precision**: {qinfo['bits']}  ")
    a(f"**Group size**: {qinfo['group_size']}  ")
    a(f"**Disk size**: {fmt(disk, ' MB')}  ")
    a(f"**Quantized by**: [{AUTHOR}](https://huggingface.co/{AUTHOR})")
    a(f"")
    a(f"## About this variant")
    a(f"")
    a(f"{qinfo['description']}")
    a(f"")
    a(f"## Benchmark results")
    a(f"")
    a(f"Evaluated on Apple M5 Pro with MLX. All metrics measured in a single pass (model loaded once).")
    a(f"")

    # perf table
    a(f"### Performance")
    a(f"")
    a(f"| | This model | FP16 baseline |")
    a(f"|---|---:|---:|")
    a(f"| Prefill (tok/s) | {fmt(p.get('avg_prefill_tps'))} | {fmt(fp.get('avg_prefill_tps'))} |")
    a(f"| Decode (tok/s)  | {fmt(p.get('avg_decode_tps'))} | {fmt(fp.get('avg_decode_tps'))} |")
    a(f"| Peak memory (GB)| {fmt(p.get('peak_memory_gb'))} | {fmt(fp.get('peak_memory_gb'))} |")
    a(f"| Disk size (MB)  | {fmt(disk)} | {fmt(disk_size_mb('fp16'))} |")
    a(f"")

    # quality table
    a(f"### Quality")
    a(f"")
    a(f"| Benchmark | This model | FP16 baseline | Task |")
    a(f"|---|---:|---:|---|")
    gsm  = b.get("gsm8k", {})
    fgsm = fb.get("gsm8k", {})
    mmlu  = b.get("mmlu", {})
    fmmlu = fb.get("mmlu", {})
    he   = b.get("humaneval", {})
    fhe  = fb.get("humaneval", {})
    a(f"| GSM8K     | {fmt(gsm.get('accuracy'), '%')} | {fmt(fgsm.get('accuracy'), '%')} | Math reasoning (25 samples) |")
    a(f"| MMLU      | {fmt(mmlu.get('accuracy'), '%')} | {fmt(fmmlu.get('accuracy'), '%')} | World knowledge (50 samples) |")
    a(f"| HumanEval | {fmt(he.get('pass_at_1'), '%')} | {fmt(fhe.get('pass_at_1'), '%')} | Code pass@1 (20 samples) |")
    a(f"")

    # context scaling
    ctx = s.get("context_scaling", []) if s else []
    if ctx:
        a(f"### Context scaling (decode tok/s)")
        a(f"")
        a(f"| Context length | Decode tok/s |")
        a(f"|---:|---:|")
        for c in ctx:
            if c.get("status") == "ok":
                a(f"| ~{c['target_tokens']} tokens | {c['generation_tps']:.1f} |")
            else:
                a(f"| ~{c['target_tokens']} tokens | OOM |")
        a(f"")

    # usage
    a(f"## Usage")
    a(f"")
    a(f"### Install")
    a(f"")
    a(f"```bash")
    a(f"pip install mlx-lm")
    a(f"```")
    a(f"")
    a(f"### Generate")
    a(f"")
    a(f"```python")
    a(f"from mlx_lm import load, generate")
    a(f"")
    a(f"model, tokenizer = load(\"{repo_name}\")")
    a(f"response = generate(model, tokenizer, prompt=\"Your prompt here\", max_tokens=512, verbose=True)")
    a(f"```")
    a(f"")
    a(f"### Stream")
    a(f"")
    a(f"```python")
    a(f"from mlx_lm import load, stream_generate")
    a(f"")
    a(f"model, tokenizer = load(\"{repo_name}\")")
    a(f"for chunk in stream_generate(model, tokenizer, prompt=\"Your prompt here\", max_tokens=512):")
    a(f"    print(chunk.text, end=\"\", flush=True)")
    a(f"```")
    a(f"")

    # all variants
    a(f"## All variants in this collection")
    a(f"")
    a(f"| Model | Method | Bits/weight |")
    a(f"|---|---|---|")
    for v_label, v_desc in ALL_VARIANTS:
        marker = " ← this model" if v_label == label else ""
        a(f"| [{AUTHOR}/granite-4.1-8b-{v_label}-mlx](https://huggingface.co/{AUTHOR}/granite-4.1-8b-{v_label}-mlx) | {v_desc} |{marker} |")
    a(f"")

    a(f"## Notes")
    a(f"")
    a(f"- Requires Apple Silicon (M1 or later) with MLX")
    a(f"- Benchmarks run on Apple M5 Pro, 24 GB unified memory")
    a(f"- Sample sizes are small (25–50 per benchmark) — treat accuracy figures as indicative, not definitive")
    a(f"- Base model license: [Apache 2.0](https://huggingface.co/{BASE_MODEL}/blob/main/LICENSE)")
    a(f"")
    a(f"## Original model")
    a(f"")
    a(f"See [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for full model details, training information, and intended use.")

    return "\n".join(lines)


if __name__ == "__main__":
    generated = []
    missing_results = []

    for label, _ in ALL_VARIANTS:
        model_dir = MODELS_DIR / f"granite-4.1-8b-{label}"
        if not model_dir.exists():
            print(f"  SKIP {label} — model folder not found")
            continue

        s = load_summary(label)
        if not s:
            missing_results.append(label)
            print(f"  WARN {label} — no benchmark results yet, card will show N/A")

        card = generate_card(label)
        out_path = model_dir / "README.md"
        with open(out_path, "w") as f:
            f.write(card)
        print(f"  ✓  {label} → {out_path}")
        generated.append(label)

    print(f"\nGenerated {len(generated)} model cards.")
    if missing_results:
        print(f"Missing results (N/A placeholders): {missing_results}")
        print("Re-run after benchmarks finish to fill in the numbers.")
