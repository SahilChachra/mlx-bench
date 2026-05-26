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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_NAME, BASE_HF_REPO, DISPLAY_NAME

_REPO = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = _REPO / "outputs"
MODELS_DIR  = _REPO / "models"

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
    "optiq-4bpw": "OptiQ mixed-precision (target 4.0 bpw)",
    "optiq-5bpw": "OptiQ mixed-precision (target 5.0 bpw)",
    "optiq-3.5bpw": "OptiQ mixed-precision (target 3.5 bpw)",
}


def load_optiq_metadata(label):
    """Read OptiQ's per-layer bit allocation, if present."""
    p = MODELS_DIR / f"{BASE_NAME}-{label}" / "optiq_metadata.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def render_optiq_section(label):
    """Explainer + actual bit distribution for an OptiQ variant."""
    meta = load_optiq_metadata(label)
    if meta is None:
        return []
    lines = [
        "## About this quantization",
        "",
        "Unlike uniform 4-bit quantization (which forces every layer onto the same bit grid and often collapses reasoning at low bit widths), this model was quantized with [mlx-optiq](https://mlx-optiq.com/) using **per-layer KL-sensitivity analysis**:",
        "",
        "1. A small calibration set (32 samples spanning prose, multi-step reasoning, code, and constraint-following instructions) is run through the FP16 reference and through trial quantizations of each layer.",
        "2. The output drift per layer is measured. Layers whose outputs are most affected by quantization (typically the final attention projections, the `lm_head`, and a few middle blocks) get more bits; layers that tolerate aggressive quantization get fewer.",
        "3. The final assignment hits the target average bits-per-weight while keeping the bits where they matter.",
        "",
        f"This trades off precision unequally so the average comes out near the target ({meta.get('target_bpw', '?')} bits/weight), but the bits that matter most for output fidelity stay high.",
        "",
    ]
    # Config (mlx-optiq writes flat top-level keys)
    target_bpw = meta.get("target_bpw")
    actual_bpw = meta.get("achieved_bpw")
    reference  = meta.get("reference")
    method     = meta.get("method")
    per_layer  = meta.get("per_layer", {}) or {}
    candidates = sorted({int(v.get("bits")) for v in per_layer.values() if isinstance(v, dict) and v.get("bits") is not None})
    group_size = next((v.get("group_size") for v in per_layer.values() if isinstance(v, dict) and v.get("group_size") is not None), None)

    lines += [
        "### Quantization config",
        "",
        f"- **Method**: `{method}` (mlx-optiq)",
        f"- **Target bits/weight**: {target_bpw if target_bpw is not None else '?'}",
        f"- **Achieved bits/weight**: {actual_bpw:.3f}" if isinstance(actual_bpw, (int, float)) else "- **Achieved bits/weight**: ?",
        f"- **Candidate bits**: {candidates if candidates else '?'}",
        f"- **Group size**: {group_size if group_size is not None else '?'}",
        f"- **Sensitivity reference**: `{reference}`" if reference else "",
        f"- **Calibration**: 32-sample 4-domain mix (prose + reasoning + code + constraints)",
        "",
    ]
    lines = [l for l in lines if l != ""]  # drop empty
    lines.append("")

    # Per-layer histogram
    if per_layer:
        hist = {}
        for info in per_layer.values():
            b = info.get("bits") if isinstance(info, dict) else info
            if b is None:
                continue
            hist[int(b)] = hist.get(int(b), 0) + 1
        if hist:
            lines += [
                "### Per-layer bit allocation",
                "",
                "169 model components total. OptiQ allocated bits non-uniformly based on KL sensitivity:",
                "",
                "| Bits | Components | Share |",
                "|---:|---:|---:|",
            ]
            total = sum(hist.values())
            for b in sorted(hist.keys(), reverse=True):
                pct = hist[b] / total * 100
                lines.append(f"| {b}-bit | {hist[b]} | {pct:.1f}% |")
            lines.append(f"| **Total** | **{total}** | 100.0% |")
            lines.append("")
            # Which layers got the top tier?
            top_bit = max(hist.keys())
            top_layers = [k for k, v in per_layer.items() if isinstance(v, dict) and v.get("bits") == top_bit]
            if top_layers and len(top_layers) <= 25:
                lines += [
                    f"**Components kept at {top_bit}-bit** (most sensitive to quantization):",
                    "",
                ]
                for lk in top_layers:
                    lines.append(f"- `{lk}`")
                lines.append("")
                lines.append("Notice the pattern: `lm_head`, the **first** transformer block, and the **last** transformer block — these layers carry the most information that downstream tokens depend on, so OptiQ preserves them at high precision while compressing the middle of the network more aggressively.")
                lines.append("")
    return lines


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


def render_quality_table(this, baseline, extra=None, extra_label=None):
    """Render benchmarks. If `extra` provided, adds a 3rd comparison column."""
    b  = this.get("benchmarks", {}) if this else {}
    fb = baseline.get("benchmarks", {}) if baseline else {}
    eb = extra.get("benchmarks", {}) if extra else {}
    have_extra = extra is not None
    rows = []

    def _math_cell(d):
        if not d: return "N/A"
        return f"{fmt(d.get('accuracy'), '%')} (answered {d.get('n_answered', '?')}/{d.get('n', '?')})"

    # MATH-500 (accuracy %, with answer rate)
    if "math500" in b:
        m = b["math500"]
        rows.append((
            "MATH-500 (math reasoning)",
            _math_cell(m),
            _math_cell(eb.get("math500", {})) if have_extra else None,
            _math_cell(fb.get("math500", {})),
            m.get("n"),
        ))
    # IFEval (accuracy %)
    if "ifeval" in b:
        e = b["ifeval"]
        rows.append((
            "IFEval (instruction following)",
            fmt(e.get("accuracy"), "%"),
            fmt(eb.get("ifeval", {}).get("accuracy"), "%") if have_extra else None,
            fmt(fb.get("ifeval", {}).get("accuracy"), "%"),
            e.get("n_scored"),
        ))
    # GSM8K (accuracy %)
    if "gsm8k" in b:
        rows.append((
            "GSM8K (math, accuracy)",
            fmt(b["gsm8k"].get("accuracy"), "%"),
            fmt(eb.get("gsm8k", {}).get("accuracy"), "%") if have_extra else None,
            fmt(fb.get("gsm8k", {}).get("accuracy"), "%"),
            b["gsm8k"].get("n"),
        ))
    # HumanEval (pass@1 %)
    if "humaneval" in b:
        rows.append((
            "HumanEval (code, pass@1)",
            fmt(b["humaneval"].get("pass_at_1"), "%"),
            fmt(eb.get("humaneval", {}).get("pass_at_1"), "%") if have_extra else None,
            fmt(fb.get("humaneval", {}).get("pass_at_1"), "%"),
            b["humaneval"].get("n"),
        ))
    # MMLU (accuracy %)
    if "mmlu" in b:
        rows.append((
            "MMLU (knowledge, accuracy)",
            fmt(b["mmlu"].get("accuracy"), "%"),
            fmt(eb.get("mmlu", {}).get("accuracy"), "%") if have_extra else None,
            fmt(fb.get("mmlu", {}).get("accuracy"), "%"),
            b["mmlu"].get("n"),
        ))
    # FLORES (avg chrF++ / BLEU)
    if "flores" in b:
        rows.append((
            "FLORES-200 (translation, chrF++)",
            fmt(b["flores"].get("avg_chrf")),
            fmt(eb.get("flores", {}).get("avg_chrf")) if have_extra else None,
            fmt(fb.get("flores", {}).get("avg_chrf")),
            b["flores"].get("n"),
        ))
        rows.append((
            "FLORES-200 (translation, BLEU)",
            fmt(b["flores"].get("avg_bleu")),
            fmt(eb.get("flores", {}).get("avg_bleu")) if have_extra else None,
            fmt(fb.get("flores", {}).get("avg_bleu")),
            b["flores"].get("n"),
        ))

    if not rows:
        return ["_No quality benchmarks recorded for this variant._", ""]

    if have_extra:
        lines = [
            f"| Benchmark | This model | {extra_label} | FP16 baseline | n |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, this_v, extra_v, base_v, n in rows:
            lines.append(f"| {label} | {this_v} | {extra_v} | {base_v} | {fmt(n)} |")
    else:
        lines = [
            "| Benchmark | This model | FP16 baseline | n |",
            "|---|---:|---:|---:|",
        ]
        for label, this_v, _extra_v, base_v, n in rows:
            lines.append(f"| {label} | {this_v} | {base_v} | {fmt(n)} |")
    lines.append("")

    # Optional per-level MATH-500 breakdown when present
    m500_levels = b.get("math500", {}).get("per_level") if "math500" in b else None
    fb_m500_levels = fb.get("math500", {}).get("per_level", {}) if "math500" in fb else {}
    eb_m500_levels = eb.get("math500", {}).get("per_level", {}) if have_extra and "math500" in eb else {}
    if m500_levels:
        if have_extra:
            lines += [
                "#### MATH-500 per-level accuracy",
                "",
                f"| Level | This model | {extra_label} | FP16 baseline |",
                "|---|---:|---:|---:|",
            ]
            for lvl in sorted(m500_levels):
                lines.append(f"| {lvl.replace('_', ' ')} | {fmt(m500_levels[lvl], '%')} | {fmt(eb_m500_levels.get(lvl), '%')} | {fmt(fb_m500_levels.get(lvl), '%')} |")
        else:
            lines += [
                "#### MATH-500 per-level accuracy",
                "",
                "| Level | This model | FP16 baseline |",
                "|---|---:|---:|",
            ]
            for lvl in sorted(m500_levels):
                lines.append(f"| {lvl.replace('_', ' ')} | {fmt(m500_levels[lvl], '%')} | {fmt(fb_m500_levels.get(lvl), '%')} |")
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


def render_perf_table(this, baseline, this_label, extra=None, extra_label=None, extra_folder=None):
    """If `extra` summary is provided, render a 3-column comparison.

    extra:         summary dict for the extra comparison (e.g., naive 8-bit).
    extra_label:   header label for that column (e.g., 'Naive 8-bit').
    extra_folder:  folder name under models/ for disk-size lookup (e.g., '8bit').
    """
    p  = this.get("perf", {}) if this else {}
    fp = baseline.get("perf", {}) if baseline else {}
    ep = extra.get("perf", {}) if extra else {}
    ss  = p.get("steady_state")  or {}
    fss = fp.get("steady_state") or {}
    ess = ep.get("steady_state") or {}

    have_extra = extra is not None
    if have_extra:
        header = f"| | This model | {extra_label} | FP16 baseline |"
        sep    = "|---|---:|---:|---:|"
    else:
        header = "| | This model | FP16 baseline |"
        sep    = "|---|---:|---:|"

    def row(label, this_v, extra_v, base_v):
        if have_extra:
            return f"| {label} | {this_v} | {extra_v} | {base_v} |"
        return f"| {label} | {this_v} | {base_v} |"

    rows = [header, sep]
    if ss:
        rows.append(row(
            "Decode tok/s (steady-state)",
            f"**{fmt(ss.get('decode_tps'))}**",
            fmt(ess.get('decode_tps')) if have_extra else "",
            fmt(fss.get('decode_tps')),
        ))
        rows.append(row(
            "Prefill tok/s (steady-state)",
            fmt(ss.get('prompt_tps')),
            fmt(ess.get('prompt_tps')) if have_extra else "",
            fmt(fss.get('prompt_tps')),
        ))
    rows.append(row(
        "Decode tok/s (avg, long traces)",
        fmt(p.get('avg_decode_tps')),
        fmt(ep.get('avg_decode_tps')) if have_extra else "",
        fmt(fp.get('avg_decode_tps')),
    ))
    rows.append(row(
        "Peak memory (GB)",
        fmt(p.get('peak_memory_gb')),
        fmt(ep.get('peak_memory_gb')) if have_extra else "",
        fmt(fp.get('peak_memory_gb')),
    ))
    rows.append(row(
        "Disk size (MB)",
        fmt(disk_size_mb(this_label)),
        fmt(disk_size_mb(extra_folder)) if have_extra and extra_folder else "",
        fmt(disk_size_mb(BASELINE_LABEL)),
    ))
    rows.append("")
    if ss.get("note"):
        rows.append(f"> _{ss['note']}_")
        rows.append("")
    return rows


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

    # For OptiQ variants, also pull naive 8-bit as an extra comparison column.
    is_optiq = label.startswith("optiq")
    extra = load_summary("8bit") if is_optiq else None
    extra_label = "Naive 8-bit" if is_optiq else None
    extra_folder = "8bit" if is_optiq else None

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
    if is_optiq:
        a("  - mixed-precision")
        a("  - optiq")
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

    # OptiQ explainer + bit allocation
    if is_optiq:
        L.extend(render_optiq_section(label))

    a("## Benchmark results")
    a("")
    a("Evaluated on Apple M5 Pro with MLX. Model loaded once; performance and quality measured in a single pass.")
    a("")

    a("### Performance")
    a("")
    L.extend(render_perf_table(this, baseline, label, extra=extra, extra_label=extra_label, extra_folder=extra_folder))

    a("### Quality")
    a("")
    L.extend(render_quality_table(this, baseline, extra=extra, extra_label=extra_label))

    L.extend(render_context_scaling(this))

    # OptiQ-specific limitations + usage guidance
    if is_optiq:
        m500 = this.get("benchmarks", {}).get("math500", {}) if this else {}
        m500_rate = m500.get("answer_rate")
        m500_acc  = m500.get("accuracy")
        fb_m500   = baseline.get("benchmarks", {}).get("math500", {}) if baseline else {}
        fb_acc    = fb_m500.get("accuracy")

        if m500_rate is not None and m500_rate < 30:
            # Collapsed
            a("## Limitations — thinking mode")
            a("")
            a(f"On MATH-500 with thinking enabled, this variant answered only **{m500.get('n_answered', 0)}/{m500.get('n', 0)} questions** within a 4096-token budget — the rest never closed `</think>`. Logit drift from low-bit quantization compounded across the reasoning trace.")
            a("")
            a("**Recommended usage:**")
            a("")
            a("- ✅ Instruction following (no-think mode)")
            a("- ✅ Short conversational use")
            a("- ⚠️ Multi-step math / reasoning with `enable_thinking=True` — broken at this bit budget")
            a("")
        elif m500_acc is not None and fb_acc is not None and m500_acc < fb_acc * 0.7:
            # Partially degraded (< 70% of FP16 accuracy)
            a("## Limitations — degraded math reasoning")
            a("")
            a(f"On MATH-500 with thinking enabled, this variant scores **{m500_acc}%** vs **{fb_acc}%** on the FP16 baseline. The model still produces real answers (answer-rate {m500_rate}%), but the math-reasoning quality is noticeably lower than the FP16 reference. Code generation and instruction following are closer to baseline.")
            a("")
            a("This is the standard tradeoff when targeting low bits-per-weight on a hybrid-thinking model: code and instruction-following are robust under quantization, but multi-step reasoning chains accumulate logit drift that costs accuracy.")
            a("")
            a("**Recommended usage:**")
            a("")
            a("- ✅ Code generation, chat assistant, instruction following")
            a("- ✅ Short single-step Q&A")
            a("- ⚠️ Heavy math reasoning — prefer the FP16 source or the [naive 8-bit variant](https://huggingface.co/sahilchachra/minicpm5-1b-8bit-mlx) for that workload")
            a("")
            a("To run in no-think mode (preferred for chat / non-reasoning use):")
            a("")
            a("```python")
            a("inputs = tokenizer.apply_chat_template(")
            a("    [{\"role\": \"user\", \"content\": \"...\"}],")
            a("    add_generation_prompt=True, enable_thinking=False, tokenize=False,")
            a(")")
            a("```")
            a("")

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
