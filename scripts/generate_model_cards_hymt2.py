"""
Generate HuggingFace model cards (README.md) for Hy-MT2-7B MLX variants.

Reads outputs/<model>/summary.json (FLORES + perf) and writes README.md
into models/<model>/.

Usage:
  python scripts/generate_model_cards_hymt2.py
"""

import json
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR  = Path(__file__).parent.parent / "models"

AUTHOR      = "sahilchachra"
BASE_NAME   = "hy-mt2-7b"
BASE_LABEL  = "fp16"
BASE_MODEL  = "tencent/Hy-MT2-7B"
DISPLAY     = "Hy-MT2-7B"

# Variants we publish — used for cross-links
ALL_VARIANTS = [
    ("4bit", "Affine int4 (group 64)"),
    ("8bit", "Affine int8 (group 64)"),
]

QUANT_DESCRIPTIONS = {
    "4bit": {
        "method": "Affine integer quantization",
        "bits": "4-bit (~4.5 bits/weight avg)",
        "group_size": 64,
        "description": (
            "Standard affine (integer) quantization at 4-bit with group size 64. "
            "Largest compression ratio — recommended when memory is tight or you "
            "want the fastest decode throughput."
        ),
    },
    "8bit": {
        "method": "Affine integer quantization",
        "bits": "8-bit (~8.5 bits/weight avg)",
        "group_size": 64,
        "description": (
            "Affine quantization at 8-bit with group size 64. "
            "Closest to FP16 translation quality. Recommended when memory allows "
            "and translation accuracy is the priority."
        ),
    },
}


def load_summary(label):
    p = OUTPUTS_DIR / f"{BASE_NAME}-{label}" / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt(v, suffix="", missing="N/A"):
    return f"{v}{suffix}" if v is not None else missing


def disk_size_mb(label):
    p = MODELS_DIR / f"{BASE_NAME}-{label}"
    if not p.exists():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**2)


def generate_card(label):
    model_name = f"{BASE_NAME}-{label}"
    repo_name  = f"{AUTHOR}/{model_name}-mlx"
    qinfo      = QUANT_DESCRIPTIONS[label]
    s          = load_summary(label)
    fp16       = load_summary(BASE_LABEL)
    disk       = disk_size_mb(label)

    b  = s.get("benchmarks", {}) if s else {}
    p  = s.get("perf", {}) if s else {}
    fp = fp16.get("perf", {}) if fp16 else {}
    fb = fp16.get("benchmarks", {}) if fp16 else {}

    lines = []
    a = lines.append

    a("---")
    a("language:")
    a("  - en")
    a("  - zh")
    a("  - fr")
    a("  - es")
    a("  - de")
    a("  - ja")
    a("  - ko")
    a("  - ar")
    a("  - ru")
    a("  - pt")
    a("  - it")
    a("license: other")
    a(f"base_model: {BASE_MODEL}")
    a("pipeline_tag: translation")
    a("tags:")
    a("  - mlx")
    a("  - quantized")
    a("  - translation")
    a("  - hunyuan")
    a("  - apple-silicon")
    a("---")
    a("")
    a(f"# {model_name}-mlx")
    a("")
    a(f"Quantized version of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for Apple Silicon using [MLX](https://github.com/ml-explore/mlx).")
    a("")
    a(f"{DISPLAY} is Tencent's multilingual translation model covering 40+ languages.")
    a("")
    a(f"**Quantization**: {qinfo['method']}  ")
    a(f"**Precision**: {qinfo['bits']}  ")
    a(f"**Group size**: {qinfo['group_size']}  ")
    a(f"**Disk size**: {fmt(disk, ' MB')}  ")
    a(f"**Quantized by**: [{AUTHOR}](https://huggingface.co/{AUTHOR})")
    a("")
    a("## About this variant")
    a("")
    a(f"{qinfo['description']}")
    a("")
    a("## Benchmark results")
    a("")
    a("Evaluated on Apple M5 Pro with MLX. Model loaded once; performance and quality measured in a single pass.")
    a("")

    a("### Performance")
    a("")
    a("| | This model | FP16 baseline |")
    a("|---|---:|---:|")
    a(f"| Prefill (tok/s)  | {fmt(p.get('avg_prefill_tps'))} | {fmt(fp.get('avg_prefill_tps'))} |")
    a(f"| Decode (tok/s)   | {fmt(p.get('avg_decode_tps'))}  | {fmt(fp.get('avg_decode_tps'))}  |")
    a(f"| Peak memory (GB) | {fmt(p.get('peak_memory_gb'))} | {fmt(fp.get('peak_memory_gb'))} |")
    a(f"| Disk size (MB)   | {fmt(disk)} | {fmt(disk_size_mb(BASE_LABEL))} |")
    a("")

    a("### Translation quality (FLORES-200 devtest)")
    a("")
    flores = b.get("flores", {})
    fflores = fb.get("flores", {})
    pairs = flores.get("per_pair", []) if flores else []
    a("Reported as **chrF++** (higher is better). Sample-size noted per pair.")
    a("")
    a("| Direction | This model | FP16 baseline | n |")
    a("|---|---:|---:|---:|")
    if pairs:
        # Build a dict of fp pair scores for alignment
        fp_pairs = {pp["pair"]: pp for pp in (fflores.get("per_pair", []) if fflores else [])}
        for pp in pairs:
            fp_pp = fp_pairs.get(pp["pair"], {})
            a(f"| {pp['pair']} | {pp.get('chrf', 'N/A')} | {fp_pp.get('chrf', 'N/A')} | {pp.get('n', 'N/A')} |")
    else:
        a("| (no FLORES results in summary.json) | N/A | N/A | N/A |")
    a("")
    a(f"**Avg chrF++**: {fmt(flores.get('avg_chrf'))} vs FP16 {fmt(fflores.get('avg_chrf'))}  ")
    a(f"**Avg BLEU**:   {fmt(flores.get('avg_bleu'))} vs FP16 {fmt(fflores.get('avg_bleu'))}")
    a("")

    ctx = s.get("context_scaling", []) if s else []
    if ctx:
        a("### Context scaling (decode tok/s)")
        a("")
        a("| Context length | Decode tok/s |")
        a("|---:|---:|")
        for c in ctx:
            if c.get("status") == "ok":
                a(f"| ~{c['target_tokens']} tokens | {c['generation_tps']:.1f} |")
            else:
                a(f"| ~{c['target_tokens']} tokens | OOM |")
        a("")

    a("## Usage")
    a("")
    a("### Install")
    a("")
    a("```bash")
    a("pip install mlx-lm")
    a("```")
    a("")
    a("### Translate")
    a("")
    a("```python")
    a("from mlx_lm import load, generate")
    a("")
    a(f'model, tokenizer = load("{repo_name}")')
    a("")
    a("prompt = (")
    a('    "Translate the following text from English to French.\\n"')
    a('    "English: The early bird catches the worm.\\n"')
    a('    "French:"')
    a(")")
    a("print(generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=True))")
    a("```")
    a("")
    a("### Stream")
    a("")
    a("```python")
    a("from mlx_lm import load, stream_generate")
    a("")
    a(f'model, tokenizer = load("{repo_name}")')
    a('for chunk in stream_generate(model, tokenizer, prompt="Translate \\"Hello world\\" to Japanese:", max_tokens=64):')
    a('    print(chunk.text, end="", flush=True)')
    a("```")
    a("")

    a("## All variants in this collection")
    a("")
    a("| Model | Method |")
    a("|---|---|")
    for v_label, v_desc in ALL_VARIANTS:
        marker = " ← this model" if v_label == label else ""
        a(f"| [{AUTHOR}/{BASE_NAME}-{v_label}-mlx](https://huggingface.co/{AUTHOR}/{BASE_NAME}-{v_label}-mlx) | {v_desc}{marker} |")
    a("")

    a("## Notes")
    a("")
    a("- Requires Apple Silicon (M1 or later) with MLX")
    a("- Benchmarks run on Apple M5 Pro, 24 GB unified memory")
    a("- FLORES-200 sample sizes are small — treat chrF/BLEU figures as indicative, not definitive")
    a(f"- License: see [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for the original model's license terms")
    a("")
    a("## Original model")
    a("")
    a(f"See [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for full model details, supported languages, and intended use.")

    return "\n".join(lines)


if __name__ == "__main__":
    generated = []
    missing_results = []

    for label, _ in ALL_VARIANTS:
        model_dir = MODELS_DIR / f"{BASE_NAME}-{label}"
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
