"""
Generate HuggingFace model cards (README.md) for Hy-MT2 MLX variants.

Works for any Hy-MT2 size (7B, 1.8B, ...). Configuration comes from env via
scripts/config.py:

  MLX_BENCH_BASE_NAME    e.g. hy-mt2-7b / hy-mt2-1.8b
  MLX_BENCH_HF_REPO      e.g. tencent/Hy-MT2-7B / tencent/Hy-MT2-1.8B
  MLX_BENCH_DISPLAY_NAME e.g. "Hy-MT2-7B" / "Hy-MT2-1.8B"

Variants are auto-discovered from models/<BASE_NAME>-<label>/ folders
(excluding fp16). Reads outputs/<model>/summary.json and writes README.md
into the model folder.

Usage:
  MLX_BENCH_BASE_NAME=hy-mt2-1.8b MLX_BENCH_HF_REPO=tencent/Hy-MT2-1.8B \
    python scripts/generate_model_cards_hymt2.py
"""

import json
import os
from pathlib import Path

from config import BASE_NAME, BASE_HF_REPO, DISPLAY_NAME

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR  = Path(__file__).parent.parent / "models"

AUTHOR      = os.environ.get("MLX_BENCH_HF_AUTHOR", "sahilchachra")
BASE_LABEL  = "fp16"

# Per-variant descriptive text — keyed by label suffix
QUANT_DESCRIPTIONS = {
    "4bit": {
        "method": "Affine integer quantization",
        "bits": "4-bit (~4.5 bits/weight avg)",
        "group_size": 64,
        "short": "Affine int4 (group 64)",
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
        "short": "Affine int8 (group 64)",
        "description": (
            "Affine quantization at 8-bit with group size 64. "
            "Closest to FP16 translation quality. Recommended when memory allows "
            "and translation accuracy is the priority."
        ),
    },
    "mxfp4": {
        "method": "Block floating-point MX FP4 (microscaling)",
        "bits": "~4 bits/weight",
        "group_size": 32,
        "short": "Block float MX FP4",
        "description": (
            "Microscaling (MX) block floating-point quantization at FP4 precision. "
            "Uses a shared floating-point exponent per block of 32 weights instead "
            "of integer affine scaling — different numerical properties vs affine int4."
        ),
    },
    "mxfp8": {
        "method": "Block floating-point MX FP8 (microscaling)",
        "bits": "~8 bits/weight",
        "group_size": 32,
        "short": "Block float MX FP8",
        "description": (
            "Microscaling (MX) block floating-point quantization at FP8 precision. "
            "Shared floating-point exponent per block of 32 weights — same bit-width "
            "as int8 but a different numerical format."
        ),
    },
    "mixed4_6": {
        "method": "Mixed-bit quantization (mlx-lm predicate: mixed_4_6)",
        "bits": "~4.77 bits/weight avg",
        "group_size": 64,
        "short": "Mixed 4+6 bit",
        "description": (
            "Mixed-bit quantization: sensitive layers (embeddings, first/last layers) "
            "at 6-bit, the rest at 4-bit. Better quality than uniform 4-bit at nearly "
            "the same size."
        ),
    },
}


def discover_variants():
    """List <label> for every models/<BASE_NAME>-<label>/ folder (excluding fp16)."""
    out = []
    if not MODELS_DIR.exists():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{BASE_NAME}-"):
            continue
        label = d.name[len(BASE_NAME) + 1:]
        if label == BASE_LABEL:
            continue
        out.append(label)
    return out


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


def generate_card(label, all_variants):
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
    a(f"base_model: {BASE_HF_REPO}")
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
    a(f"Quantized version of [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO}) for Apple Silicon using [MLX](https://github.com/ml-explore/mlx).")
    a("")
    a(f"{DISPLAY_NAME} is Tencent's multilingual translation model covering 40+ languages.")
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
    for v_label in all_variants:
        v_desc = QUANT_DESCRIPTIONS.get(v_label, {}).get("short", v_label)
        marker = " ← this model" if v_label == label else ""
        a(f"| [{AUTHOR}/{BASE_NAME}-{v_label}-mlx](https://huggingface.co/{AUTHOR}/{BASE_NAME}-{v_label}-mlx) | {v_desc}{marker} |")
    a("")

    a("## Notes")
    a("")
    a("- Requires Apple Silicon (M1 or later) with MLX")
    a("- Benchmarks run on Apple M5 Pro, 24 GB unified memory")
    a("- FLORES-200 sample sizes are small — treat chrF/BLEU figures as indicative, not definitive")
    a(f"- License: see [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO}) for the original model's license terms")
    a("")
    a("## Original model")
    a("")
    a(f"See [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO}) for full model details, supported languages, and intended use.")

    return "\n".join(lines)


if __name__ == "__main__":
    all_variants = discover_variants()
    if not all_variants:
        print(f"No quantized variants found under models/{BASE_NAME}-*")
        raise SystemExit(1)

    print(f"Base name : {BASE_NAME}")
    print(f"HF repo   : {BASE_HF_REPO}")
    print(f"Variants  : {all_variants}")
    print()

    generated = []
    missing_results = []

    for label in all_variants:
        model_dir = MODELS_DIR / f"{BASE_NAME}-{label}"
        if label not in QUANT_DESCRIPTIONS:
            print(f"  WARN {label} — no QUANT_DESCRIPTIONS entry; using generic copy")
            QUANT_DESCRIPTIONS[label] = {
                "method": f"Quantization ({label})",
                "bits": label,
                "group_size": "?",
                "short": label,
                "description": f"Quantized variant: {label}.",
            }

        s = load_summary(label)
        if not s:
            missing_results.append(label)
            print(f"  WARN {label} — no benchmark results yet, card will show N/A")

        card = generate_card(label, all_variants)
        out_path = model_dir / "README.md"
        with open(out_path, "w") as f:
            f.write(card)
        print(f"  ✓  {label} → {out_path}")
        generated.append(label)

    print(f"\nGenerated {len(generated)} model cards.")
    if missing_results:
        print(f"Missing results (N/A placeholders): {missing_results}")
