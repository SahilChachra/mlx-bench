"""
Compile all benchmark results into a comparison report.

Generic: auto-discovers which benchmarks each summary contains and only renders
the columns that have data. Works for any model that the pipeline has run on.

Usage:
  python scripts/generate_report.py
  python scripts/generate_report.py --out reports/my_report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_NAME, BASE_HF_REPO, DISPLAY_NAME

_REPO = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = _REPO / "outputs"
REPORTS_DIR = _REPO / "reports"
MODELS_DIR  = _REPO / "models"
REPORTS_DIR.mkdir(exist_ok=True)

SORT_ORDER = ["fp16", "8bit", "mxfp8", "6bit", "5bit", "mixed4_6", "4bit", "mxfp4"]


def load_summary(model_dir):
    p = model_dir / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def disk_size_mb(model_name):
    p = MODELS_DIR / model_name
    if not p.exists():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**2)


def discover():
    if not OUTPUTS_DIR.exists():
        return []
    dirs = [d for d in OUTPUTS_DIR.iterdir()
            if d.is_dir() and d.name.startswith(f"{BASE_NAME}-") and (d / "summary.json").exists()]

    def sort_key(d):
        for i, k in enumerate(SORT_ORDER):
            if k in d.name:
                return i
        return 99

    return sorted(dirs, key=sort_key)


def fmt(v, suffix="", missing="—"):
    return f"{v}{suffix}" if v is not None else missing


def collect_quality_columns(summaries):
    """Return list of (column_label, benchmark_key, field, suffix) found in any summary."""
    seen = []
    candidates = [
        ("GSM8K",         "gsm8k",     "accuracy", "%"),
        ("MMLU",          "mmlu",      "accuracy", "%"),
        ("HumanEval",     "humaneval", "pass_at_1", "%"),
        ("FLORES chrF++", "flores",    "avg_chrf", ""),
        ("FLORES BLEU",   "flores",    "avg_bleu", ""),
    ]
    for col, bm, field, suffix in candidates:
        for _, s in summaries:
            b = s.get("benchmarks", {}).get(bm, {})
            if field in b:
                seen.append((col, bm, field, suffix))
                break
    return seen


def generate_report(out_path):
    model_dirs = discover()
    if not model_dirs:
        print(f"No results found in outputs/{BASE_NAME}-*. Run the benchmark script first.")
        return

    summaries = [(d, load_summary(d)) for d in model_dirs]
    summaries = [(d, s) for d, s in summaries if s]
    labels = [s["label"] for _, s in summaries]
    print(f"Building report for: {labels}")

    quality_cols = collect_quality_columns(summaries)

    L = []
    a = L.append

    a(f"# {DISPLAY_NAME} — Quantization Benchmark Report")
    a("")
    a(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    a("**Hardware**: Apple M5 Pro (MLX)  ")
    a(f"**Base model**: [{BASE_HF_REPO}](https://huggingface.co/{BASE_HF_REPO})")
    a("")
    a("---")
    a("")

    # ── Summary table ─────────────────────────────────────────────────────────
    a("## Summary")
    a("")
    header = ["Model", "Disk (MB)", "Prefill tok/s", "Decode tok/s", "Peak Mem (GB)"]
    align  = ["",      "---:",      "---:",          "---:",         "---:"]
    for col, *_ in quality_cols:
        header.append(col)
        align.append("---:")

    a("| " + " | ".join(header) + " |")
    a("|" + "|".join((["---"] if x == "" else [x]) for x in align).replace("[", "").replace("]", "").replace("'", "") + "|")
    # Cleaner separator row:
    L[-1] = "|" + "|".join((c if c else "---") for c in align) + "|"

    for d, s in summaries:
        b = s.get("benchmarks", {})
        p = s.get("perf", {})
        row = [
            s["label"],
            fmt(disk_size_mb(d.name)),
            fmt(p.get("avg_prefill_tps")),
            fmt(p.get("avg_decode_tps")),
            fmt(p.get("peak_memory_gb")),
        ]
        for _, bm, field, suffix in quality_cols:
            row.append(fmt(b.get(bm, {}).get(field), suffix))
        a("| " + " | ".join(str(x) for x in row) + " |")

    a("")
    a("---")
    a("")

    # ── Context scaling ───────────────────────────────────────────────────────
    a("## Context scaling (decode tok/s)")
    a("")
    a("| Model | ~128 tok | ~256 tok | ~512 tok | ~1024 tok |")
    a("|---|---:|---:|---:|---:|")
    for d, s in summaries:
        ctx = s.get("context_scaling", [])
        vals = []
        for c in ctx:
            if c.get("status") == "ok":
                vals.append(f"{c['generation_tps']:.1f}")
            else:
                vals.append("OOM")
        while len(vals) < 4:
            vals.append("—")
        a(f"| {s['label']} | {' | '.join(vals[:4])} |")
    a("")
    a("---")
    a("")

    # ── Per-benchmark detail ─────────────────────────────────────────────────
    benchmarks_present = set()
    for _, s in summaries:
        benchmarks_present.update(s.get("benchmarks", {}).keys())

    if "gsm8k" in benchmarks_present or "mmlu" in benchmarks_present or "humaneval" in benchmarks_present:
        a("## Quality detail")
        a("")
        if "gsm8k" in benchmarks_present:
            a("### GSM8K")
            a("")
            a("| Model | Accuracy | n |")
            a("|---|---:|---:|")
            for d, s in summaries:
                g = s.get("benchmarks", {}).get("gsm8k", {})
                a(f"| {s['label']} | {fmt(g.get('accuracy'), '%')} | {fmt(g.get('n'))} |")
            a("")
        if "mmlu" in benchmarks_present:
            a("### MMLU")
            a("")
            a("| Model | Accuracy | n |")
            a("|---|---:|---:|")
            for d, s in summaries:
                g = s.get("benchmarks", {}).get("mmlu", {})
                a(f"| {s['label']} | {fmt(g.get('accuracy'), '%')} | {fmt(g.get('n'))} |")
            a("")
        if "humaneval" in benchmarks_present:
            a("### HumanEval")
            a("")
            a("| Model | pass@1 | Syntax OK | n |")
            a("|---|---:|---:|---:|")
            for d, s in summaries:
                g = s.get("benchmarks", {}).get("humaneval", {})
                a(f"| {s['label']} | {fmt(g.get('pass_at_1'), '%')} | {fmt(g.get('syntax_rate'), '%')} | {fmt(g.get('n'))} |")
            a("")
        a("---")
        a("")

    if "flores" in benchmarks_present:
        a("## FLORES-200 translation quality")
        a("")
        a("| Model | Avg chrF++ | Avg BLEU | n |")
        a("|---|---:|---:|---:|")
        for d, s in summaries:
            f_ = s.get("benchmarks", {}).get("flores", {})
            a(f"| {s['label']} | {fmt(f_.get('avg_chrf'))} | {fmt(f_.get('avg_bleu'))} | {fmt(f_.get('n'))} |")
        a("")

        # Per-pair breakdown — collect union of pairs across summaries
        all_pairs = []
        seen = set()
        for _, s in summaries:
            for pp in s.get("benchmarks", {}).get("flores", {}).get("per_pair", []) or []:
                if pp["pair"] not in seen:
                    seen.add(pp["pair"])
                    all_pairs.append(pp["pair"])
        if all_pairs:
            a("### chrF++ per direction")
            a("")
            a("| Model | " + " | ".join(all_pairs) + " |")
            a("|---|" + "|".join(["---:"] * len(all_pairs)) + "|")
            for d, s in summaries:
                per = {pp["pair"]: pp for pp in s.get("benchmarks", {}).get("flores", {}).get("per_pair", []) or []}
                row = [s["label"]]
                for pair in all_pairs:
                    row.append(fmt(per.get(pair, {}).get("chrf")))
                a("| " + " | ".join(row) + " |")
            a("")
        a("---")
        a("")

    report = "\n".join(L)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS_DIR / f"{BASE_NAME}_benchmark.md"))
    args = parser.parse_args()
    generate_report(Path(args.out))
