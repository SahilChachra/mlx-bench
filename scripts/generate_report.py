"""
Compile all benchmark results into a final comparison report.

Usage:
  python scripts/generate_report.py
  python scripts/generate_report.py --out reports/my_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
MODELS_DIR  = Path(__file__).parent.parent / "models"
REPORTS_DIR.mkdir(exist_ok=True)

SORT_ORDER = ["fp16", "8bit", "6bit", "5bit", "mixed4_6", "4bit"]


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
    dirs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and (d / "summary.json").exists()]

    def sort_key(d):
        for i, k in enumerate(SORT_ORDER):
            if k in d.name:
                return i
        return 99

    return sorted(dirs, key=sort_key)


def fmt(v, suffix="", missing="—"):
    return f"{v}{suffix}" if v is not None else missing


def generate_report(out_path):
    model_dirs = discover()
    if not model_dirs:
        print("No results found in outputs/. Run benchmark.py first.")
        return

    summaries = [(d, load_summary(d)) for d in model_dirs]
    summaries = [(d, s) for d, s in summaries if s]
    labels = [s["label"] for _, s in summaries]
    print(f"Building report for: {labels}")

    lines = []
    a = lines.append

    a(f"# Granite 4.1 8B — Quantization Benchmark Report")
    a(f"")
    a(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a(f"**Hardware**: Apple M5 Pro (MLX)")
    a(f"**Base model**: ibm-granite/granite-4.1-8b")
    a(f"")
    a(f"---")
    a(f"")

    # ── Summary table ──────────────────────────────────────────────────────────
    a(f"## Summary")
    a(f"")
    a(f"| Model | Disk (MB) | Prefill tok/s | Decode tok/s | Peak Mem (GB) | GSM8K | MMLU | HumanEval |")
    a(f"|---|---:|---:|---:|---:|---:|---:|---:|")

    for d, s in summaries:
        b  = s.get("benchmarks", {})
        p  = s.get("perf", {})
        disk = fmt(disk_size_mb(d.name))
        a(
            f"| {s['label']}"
            f" | {disk}"
            f" | {fmt(p.get('avg_prefill_tps'))}"
            f" | {fmt(p.get('avg_decode_tps'))}"
            f" | {fmt(p.get('peak_memory_gb'))}"
            f" | {fmt(b.get('gsm8k', {}).get('accuracy'), '%')}"
            f" | {fmt(b.get('mmlu', {}).get('accuracy'), '%')}"
            f" | {fmt(b.get('humaneval', {}).get('pass_at_1'), '%')}"
            f" |"
        )

    a(f"")
    a(f"---")
    a(f"")

    # ── Performance ────────────────────────────────────────────────────────────
    a(f"## Performance")
    a(f"")
    a(f"### Decode throughput per benchmark")
    a(f"")
    a(f"| Model | GSM8K tok/s | HumanEval tok/s | MMLU tok/s | Long-ctx tok/s |")
    a(f"|---|---:|---:|---:|---:|")

    for d, s in summaries:
        def avg_tps(name):
            records = load_jsonl(d / f"{name}.jsonl")
            if not records:
                return None
            return round(sum(r["generation_tps"] for r in records) / len(records), 1)

        a(
            f"| {s['label']}"
            f" | {fmt(avg_tps('gsm8k'))}"
            f" | {fmt(avg_tps('humaneval'))}"
            f" | {fmt(avg_tps('mmlu'))}"
            f" | {fmt(avg_tps('long_context'))}"
            f" |"
        )

    a(f"")
    a(f"### Context scaling (decode tok/s)")
    a(f"")
    a(f"| Model | ~128 tok | ~256 tok | ~512 tok | ~1024 tok |")
    a(f"|---|---:|---:|---:|---:|")

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

    a(f"")
    a(f"---")
    a(f"")

    # ── Quality ────────────────────────────────────────────────────────────────
    a(f"## Quality")
    a(f"")
    a(f"### GSM8K — Math reasoning")
    a(f"")
    a(f"| Model | Accuracy | Correct | Total |")
    a(f"|---|---:|---:|---:|")

    for d, s in summaries:
        g = s.get("benchmarks", {}).get("gsm8k", {})
        n = g.get("n", "—")
        acc = g.get("accuracy")
        correct = round(acc * n / 100) if acc and isinstance(n, int) else "—"
        a(f"| {s['label']} | {fmt(acc, '%')} | {correct} | {n} |")

    a(f"")
    a(f"### MMLU — World knowledge")
    a(f"")
    a(f"| Model | Accuracy | Correct | Total |")
    a(f"|---|---:|---:|---:|")

    for d, s in summaries:
        g = s.get("benchmarks", {}).get("mmlu", {})
        n = g.get("n", "—")
        acc = g.get("accuracy")
        correct = round(acc * n / 100) if acc and isinstance(n, int) else "—"
        a(f"| {s['label']} | {fmt(acc, '%')} | {correct} | {n} |")

    a(f"")
    a(f"### HumanEval — Code generation")
    a(f"")
    a(f"| Model | pass@1 | Syntax OK | Total |")
    a(f"|---|---:|---:|---:|")

    for d, s in summaries:
        g = s.get("benchmarks", {}).get("humaneval", {})
        a(
            f"| {s['label']}"
            f" | {fmt(g.get('pass_at_1'), '%')}"
            f" | {fmt(g.get('syntax_rate'), '%')}"
            f" | {fmt(g.get('n'))}"
            f" |"
        )

    a(f"")
    a(f"---")
    a(f"")

    # ── Failure analysis ───────────────────────────────────────────────────────
    a(f"## Failure Analysis")
    a(f"")
    a(f"### GSM8K failures")
    a(f"")

    for d, s in summaries:
        failures = [r for r in load_jsonl(d / "gsm8k.jsonl") if not r.get("correct")]
        if not failures:
            continue
        a(f"**{s['label']}** — {len(failures)} failures")
        a(f"")
        for r in failures[:3]:
            a(f"- Q: {r['question'][:80]}...")
            a(f"  Gold: `{r['gold']}` | Predicted: `{r['predicted']}`")
        if len(failures) > 3:
            a(f"- *(+{len(failures)-3} more)*")
        a(f"")

    a(f"### HumanEval failures")
    a(f"")

    for d, s in summaries:
        failures = [r for r in load_jsonl(d / "humaneval.jsonl") if not r.get("tests_passed")]
        if not failures:
            continue
        a(f"**{s['label']}** — {len(failures)} failures")
        a(f"")
        for r in failures[:3]:
            a(f"- {r['task_id']}: {r.get('error', '—')[:80]}")
        if len(failures) > 3:
            a(f"- *(+{len(failures)-3} more)*")
        a(f"")

    a(f"---")
    a(f"")

    # ── Recommendations ────────────────────────────────────────────────────────
    a(f"## Recommendations")
    a(f"")
    a(f"| Use Case | Recommended | Reason |")
    a(f"|---|---|---|")
    a(f"| Max quality | 8bit | Closest to FP16 |")
    a(f"| Best balance | 6bit | Good quality, ~2.5× faster than FP16 |")
    a(f"| RAM-constrained | mixed4_6 | Better quality than uniform 4bit, similar size |")
    a(f"| Minimum footprint | 4bit | Smallest disk + memory |")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS_DIR / "final_benchmark.md"))
    args = parser.parse_args()
    generate_report(Path(args.out))
