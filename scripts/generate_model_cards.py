"""
Generate generic HuggingFace model cards (README.md) for quantized variants.

Reads from outputs/<model>/summary.json and writes README.md into models/<model>/.

The card is intentionally **task-agnostic**: it surfaces whatever benchmarks
appear in the summary (e.g. gsm8k / humaneval / mmlu for reasoning models, or
flores for translation models). For task-specific cards with curated copy,
write a dedicated `generate_model_cards_<model>.py` alongside this one.

Configuration is read from environment via `config.py`:
  MLX_BENCH_BASE_NAME   e.g. granite-4.1-8b  / hy-mt2-7b
  MLX_BENCH_HF_REPO     e.g. ibm-granite/granite-4.1-8b
  MLX_BENCH_DISPLAY_NAME (optional, defaults to BASE_NAME)
  MLX_BENCH_HF_AUTHOR   HF username for the published repos (default: sahilchachra)
  MLX_BENCH_LICENSE     SPDX-style license tag for the card frontmatter (default: apache-2.0)

Usage:
  MLX_BENCH_BASE_NAME=granite-4.1-8b python scripts/generate_model_cards.py
"""

import json
import os
from pathlib import Path

from config import BASE_NAME, BASE_HF_REPO, DISPLAY_NAME

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR  = Path(__file__).parent.parent / "models"

AUTHOR  = os.environ.get("MLX_BENCH_HF_AUTHOR", "sahilchachra")
LICENSE = os.environ.get("MLX_BENCH_LICENSE", "apache-2.0")

# Auto-discover which variants we have. fp16 (baseline) is skipped — we don't
# republish the original — but is used as the comparison column.
BASELINE_LABEL = "fp16"


# Cross-link descriptions for common variants. Anything not in this map
# falls back to a generic label like "Quantized variant: 5bit".
VARIANT_DESCRIPTIONS = {
    "4bit":     "Affine int4",
    "5bit":     "Affine int5",
    "6bit":     "Affine int6",
    "8bit":     "Affine int8",
    "mixed4_6": "Mixed 4+6 bit",
    "mixed3_6": "Mixed 3+6 bit",
    "mxfp4":    "Block float MX FP4",
    "mxfp8":    "Block float MX FP8",
    "nvfp4":    "Block float NV FP4",
}


def discover_variants():
    """Find <BASE_NAME>-<label>/ folders in models/, excluding the baseline."""
    if not MODELS_DIR.exists():
        return []
    variants = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{BASE_NAME}-"):
            continue
        label = d.name[len(BASE_NAME) + 1:]
        if label == BASELINE_LABEL:
            continue
        variants.append(label)
    return variants


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


def render_quality_table(this, baseline):
    """Render the benchmarks that exist in summary["benchmarks"]."""
    b  = this.get("benchmarks", {}) if this else {}
    fb = baseline.get("benchmarks", {}) if baseline else {}
    rows = []

    # GSM8K (accuracy %)
    if "gsm8k" in b:
        rows.append((
            "GSM8K (math, accuracy)",
            fmt(b["gsm8k"].get("accuracy"), "%"),
            fmt(fb.get("gsm8k", {}).get("accuracy"), "%"),
            b["gsm8k"].get("n"),
        ))
    # HumanEval (pass@1 %)
    if "humaneval" in b:
        rows.append((
            "HumanEval (code, pass@1)",
            fmt(b["humaneval"].get("pass_at_1"), "%"),
            fmt(fb.get("humaneval", {}).get("pass_at_1"), "%"),
            b["humaneval"].get("n"),
        ))
    # MMLU (accuracy %)
    if "mmlu" in b:
        rows.append((
            "MMLU (knowledge, accuracy)",
            fmt(b["mmlu"].get("accuracy"), "%"),
            fmt(fb.get("mmlu", {}).get("accuracy"), "%"),
            b["mmlu"].get("n"),
        ))
    # FLORES (avg chrF++ / BLEU)
    if "flores" in b:
        rows.append((
            "FLORES-200 (translation, chrF++)",
            fmt(b["flores"].get("avg_chrf")),
            fmt(fb.get("flores", {}).get("avg_chrf")),
            b["flores"].get("n"),
        ))
        rows.append((
            "FLORES-200 (translation, BLEU)",
            fmt(b["flores"].get("avg_bleu")),
            fmt(fb.get("flores", {}).get("avg_bleu")),
            b["flores"].get("n"),
        ))

    if not rows:
        return ["_No quality benchmarks recorded for this variant._", ""]

    lines = [
        "| Benchmark | This model | FP16 baseline | n |",
        "|---|---:|---:|---:|",
    ]
    for label, this_v, base_v, n in rows:
        lines.append(f"| {label} | {this_v} | {base_v} | {fmt(n)} |")
    lines.append("")

    # Optional per-pair FLORES breakdown when present
    per_pair = b.get("flores", {}).get("per_pair") if "flores" in b else None
    fb_pairs = {p["pair"]: p for p in fb.get("flores", {}).get("per_pair", [])} if "flores" in fb else {}
    if per_pair:
        lines += [
            "#### FLORES-200 per-pair chrF++",
            "",
            "| Direction | This model | FP16 baseline | n |",
            "|---|---:|---:|---:|",
        ]
        for p in per_pair:
            base_chrf = fb_pairs.get(p["pair"], {}).get("chrf")
            lines.append(f"| {p['pair']} | {fmt(p.get('chrf'))} | {fmt(base_chrf)} | {fmt(p.get('n'))} |")
        lines.append("")

    return lines


def render_perf_table(this, baseline, this_label):
    p  = this.get("perf", {}) if this else {}
    fp = baseline.get("perf", {}) if baseline else {}
    return [
        "| | This model | FP16 baseline |",
        "|---|---:|---:|",
        f"| Prefill (tok/s)  | {fmt(p.get('avg_prefill_tps'))} | {fmt(fp.get('avg_prefill_tps'))} |",
        f"| Decode (tok/s)   | {fmt(p.get('avg_decode_tps'))}  | {fmt(fp.get('avg_decode_tps'))}  |",
        f"| Peak memory (GB) | {fmt(p.get('peak_memory_gb'))} | {fmt(fp.get('peak_memory_gb'))} |",
        f"| Disk size (MB)   | {fmt(disk_size_mb(this_label))} | {fmt(disk_size_mb(BASELINE_LABEL))} |",
        "",
    ]


def render_context_scaling(this):
    ctx = this.get("context_scaling", []) if this else []
    if not ctx:
        return []
    lines = ["### Context scaling (decode tok/s)", "",
             "| Context length | Decode tok/s |",
             "|---:|---:|"]
    for c in ctx:
        if c.get("status") == "ok":
            lines.append(f"| ~{c['target_tokens']} tokens | {c['generation_tps']:.1f} |")
        else:
            lines.append(f"| ~{c['target_tokens']} tokens | OOM |")
    lines.append("")
    return lines


def render_card(label, all_variants):
    model_name = f"{BASE_NAME}-{label}"
    repo_name  = f"{AUTHOR}/{model_name}-mlx"
    this       = load_summary(label)
    baseline   = load_summary(BASELINE_LABEL)
    disk       = disk_size_mb(label)
    variant_desc = VARIANT_DESCRIPTIONS.get(label, f"Quantized variant: {label}")

    L = []
    a = L.append

    # Frontmatter
    a("---")
    a(f"license: {LICENSE}")
    a(f"base_model: {BASE_HF_REPO}")
    a("tags:")
    a("  - mlx")
    a("  - quantized")
    a("  - apple-silicon")
    a("---")
    a("")
    a(f"# {model_name}-mlx")
    a("")
    a(f"MLX quantization of [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO}) for Apple Silicon.")
    a("")
    a(f"**Variant**: {variant_desc}  ")
    a(f"**Disk size**: {fmt(disk, ' MB')}  ")
    a(f"**Quantized by**: [{AUTHOR}](https://huggingface.co/{AUTHOR})")
    a("")

    a("## Benchmark results")
    a("")
    a("Evaluated on Apple M5 Pro with MLX. Model loaded once; performance and quality measured in a single pass.")
    a("")

    a("### Performance")
    a("")
    L.extend(render_perf_table(this, baseline, label))

    a("### Quality")
    a("")
    L.extend(render_quality_table(this, baseline))

    L.extend(render_context_scaling(this))

    # Usage
    a("## Usage")
    a("")
    a("```bash")
    a("pip install mlx-lm")
    a("```")
    a("")
    a("```python")
    a("from mlx_lm import load, generate")
    a("")
    a(f'model, tokenizer = load("{repo_name}")')
    a('response = generate(model, tokenizer, prompt="Your prompt here", max_tokens=256, verbose=True)')
    a("```")
    a("")

    # Cross-links
    if all_variants:
        a("## All variants in this collection")
        a("")
        a("| Model | Variant |")
        a("|---|---|")
        for v in all_variants:
            desc = VARIANT_DESCRIPTIONS.get(v, v)
            marker = " ← this model" if v == label else ""
            a(f"| [{AUTHOR}/{BASE_NAME}-{v}-mlx](https://huggingface.co/{AUTHOR}/{BASE_NAME}-{v}-mlx) | {desc}{marker} |")
        a("")

    a("## Notes")
    a("")
    a("- Requires Apple Silicon (M1 or later) with MLX")
    a("- Benchmarks run on Apple M5 Pro, 24 GB unified memory")
    a(f"- License: see [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO}) for the original model's license")
    a("")
    a("## Original model")
    a("")
    a(f"See [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO}) for full model details and intended use.")

    return "\n".join(L)


if __name__ == "__main__":
    variants = discover_variants()
    if not variants:
        print(f"No quantized variants found under models/{BASE_NAME}-*")
        raise SystemExit(1)

    print(f"Base name : {BASE_NAME}")
    print(f"Base repo : {BASE_HF_REPO}")
    print(f"Variants  : {variants}")
    print()

    written = []
    for label in variants:
        model_dir = MODELS_DIR / f"{BASE_NAME}-{label}"
        card = render_card(label, variants)
        out = model_dir / "README.md"
        with open(out, "w") as f:
            f.write(card)
        print(f"  ✓ {label} → {out}")
        written.append(label)

    print(f"\nGenerated {len(written)} model cards.")
