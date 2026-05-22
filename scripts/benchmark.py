"""
Combined performance + quality benchmark.

Loads the model once, runs all datasets, captures tok/s and accuracy together.

Usage:
  python scripts/benchmark.py --model ./models/granite-4.1-8b-fp16 --label fp16
  python scripts/benchmark.py --all
"""

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUTS_DIR  = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CONTEXT_LENGTHS = [128, 256, 512, 1024]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(model, tokenizer, prompt, max_tokens=512):
    """Call stream_generate, return (response_text, final GenerationResponse)."""
    chunks = []
    last = None
    n = 0
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        chunks.append(resp.text)
        last = resp
        n += 1
        if n % 30 == 0:
            print(f"      ... {n} tokens | decode {resp.generation_tps:.1f} tok/s", end="\r", flush=True)
    if n >= 30:
        print()
    if last is None:
        raise RuntimeError("Model generated 0 tokens — check model load and prompt.")
    return "".join(chunks), last


def perf_fields(resp):
    return {
        "prompt_tokens":    resp.prompt_tokens,
        "generation_tokens": resp.generation_tokens,
        "prompt_tps":       round(resp.prompt_tps, 2),
        "generation_tps":   round(resp.generation_tps, 2),
        "peak_memory_gb":   round(resp.peak_memory, 3),
        "finish_reason":    resp.finish_reason,
    }


# ── GSM8K ─────────────────────────────────────────────────────────────────────

GSM8K_PROMPT = """\
Solve the following math problem step by step. At the end, write your final answer as:
ANSWER: <number>

Problem: {question}"""


def extract_gsm8k_answer(text):
    m = re.search(r"ANSWER:\s*([\d,\.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def extract_gsm8k_gold(answer_text):
    m = re.search(r"####\s*([\d,]+)", answer_text)
    if m:
        return m.group(1).replace(",", "").strip()
    return answer_text.strip().split("\n")[-1].replace(",", "").strip()


def run_gsm8k(model, tokenizer, label):
    path = DATASETS_DIR / "gsm8k.jsonl"
    samples = load_jsonl(path)
    print(f"\n[GSM8K] {len(samples)} samples — math reasoning accuracy")
    results = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['question'][:80]}...")
        prompt = GSM8K_PROMPT.format(question=s["question"])
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=2048)

        predicted = extract_gsm8k_answer(text)
        gold = extract_gsm8k_gold(s["answer"])
        correct = predicted == gold

        results.append({
            "benchmark": "gsm8k", "model_label": label,
            "question": s["question"], "gold": gold,
            "predicted": predicted, "correct": correct,
            "response": text,
            **perf_fields(resp),
        })
        mark = "✓" if correct else "✗"
        print(f"      {mark} pred={predicted} gold={gold} | decode {resp.generation_tps:.1f} tok/s | peak {resp.peak_memory:.2f} GB")

    acc = sum(r["correct"] for r in results) / len(results) * 100
    print(f"  GSM8K accuracy: {acc:.1f}%  ({sum(r['correct'] for r in results)}/{len(results)})")
    return results, {"accuracy": round(acc, 1), "n": len(results)}


# ── HumanEval ─────────────────────────────────────────────────────────────────

HUMANEVAL_PROMPT = "{prompt}"


def _extract_code(prompt, response):
    """Extract the function body from model response."""
    # Strip echoed prompt
    if response.startswith(prompt.strip()):
        response = response[len(prompt.strip()):]

    # Prefer explicit ```python block
    md = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if md:
        text = md.group(1)
    else:
        response = re.sub(r"```\w*\s*$", "", response.rstrip(), flags=re.MULTILINE).rstrip()
        # Cut at unindented top-level statements that signal end of the function body
        cut = re.search(r"\n(?=def |class |if __name__)", response)
        if cut:
            response = response[:cut.start()]
        text = response

    # Fix indentation: use the first non-empty line as the indent reference,
    # remap it to 4 spaces, and shift all other lines proportionally
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    base = len(non_empty[0]) - len(non_empty[0].lstrip())
    result = []
    for l in lines:
        if l.strip():
            current = len(l) - len(l.lstrip())
            relative = max(0, current - base)
            result.append("    " + " " * relative + l.lstrip())
        else:
            result.append("")
    return "\n".join(result)


def run_humaneval(model, tokenizer, label):
    path = DATASETS_DIR / "humaneval.jsonl"
    samples = load_jsonl(path)
    print(f"\n[HumanEval] {len(samples)} samples — code pass@1")
    results = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['task_id']} — {s['entry_point']}")
        mx.clear_cache()
        text, resp = run(model, tokenizer, HUMANEVAL_PROMPT.format(prompt=s["prompt"]), max_tokens=4096)

        code = _extract_code(s["prompt"], text)

        try:
            compile(s["prompt"] + "\n" + code, "<string>", "exec")
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        passed, error_msg = False, ""
        if syntax_ok:
            try:
                exec(compile(s["prompt"] + "\n" + code + "\n" + s["test"] + f"\ncheck({s['entry_point']})", "<string>", "exec"), {})
                passed = True
            except Exception as e:
                error_msg = str(e)[:100]

        results.append({
            "benchmark": "humaneval", "model_label": label,
            "task_id": s["task_id"], "entry_point": s["entry_point"],
            "syntax_ok": syntax_ok, "tests_passed": passed, "error": error_msg,
            "generated_code": code,
            **perf_fields(resp),
        })
        mark = "✓" if passed else ("~" if syntax_ok else "✗")
        print(f"      {mark} {'passed' if passed else error_msg[:60]} | decode {resp.generation_tps:.1f} tok/s")

    pass_at_1 = sum(r["tests_passed"] for r in results) / len(results) * 100
    syntax_rate = sum(r["syntax_ok"] for r in results) / len(results) * 100
    print(f"  HumanEval pass@1: {pass_at_1:.1f}%  syntax ok: {syntax_rate:.1f}%")
    return results, {"pass_at_1": round(pass_at_1, 1), "syntax_rate": round(syntax_rate, 1), "n": len(results)}


# ── MMLU ──────────────────────────────────────────────────────────────────────

MMLU_PROMPT = """\
Answer the following multiple choice question. Reply with only the letter A, B, C, or D.

Question: {question}
A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""

CHOICES = ["A", "B", "C", "D"]


def run_mmlu(model, tokenizer, label):
    path = DATASETS_DIR / "mmlu.jsonl"
    samples = load_jsonl(path)
    print(f"\n[MMLU] {len(samples)} samples — multiple choice accuracy")
    results = []

    for i, s in enumerate(samples):
        choices = (s["choices"] + ["", "", "", ""])[:4]
        prompt = MMLU_PROMPT.format(
            question=s["question"], a=choices[0], b=choices[1], c=choices[2], d=choices[3]
        )
        print(f"  [{i+1}/{len(samples)}] [{s['subject']}] {s['question'][:70]}...")
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=10)

        resp_clean = text.strip().upper()
        predicted = resp_clean[0] if resp_clean and resp_clean[0] in CHOICES else None
        gold = CHOICES[s["answer"]] if s["answer"] < 4 else None
        correct = predicted == gold

        results.append({
            "benchmark": "mmlu", "model_label": label,
            "subject": s["subject"], "question": s["question"],
            "gold": gold, "predicted": predicted,
            "raw_response": text[:30], "correct": correct,
            **perf_fields(resp),
        })
        mark = "✓" if correct else "✗"
        print(f"      {mark} pred={predicted} gold={gold}")

    acc = sum(r["correct"] for r in results) / len(results) * 100
    print(f"  MMLU accuracy: {acc:.1f}%  ({sum(r['correct'] for r in results)}/{len(results)})")
    return results, {"accuracy": round(acc, 1), "n": len(results)}


# ── Long-context ──────────────────────────────────────────────────────────────

def run_long_context(model, tokenizer, label):
    path = DATASETS_DIR / "long_context_prompts.jsonl"
    samples = load_jsonl(path)
    print(f"\n[Long-context] {len(samples)} samples — sustained generation")
    results = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['id']} — {s['prompt'][:70]}...")
        mx.clear_cache()
        text, resp = run(model, tokenizer, s["prompt"], max_tokens=2048)

        min_tokens = s.get("expected_min_tokens", 300)
        met = resp.generation_tokens >= min_tokens

        results.append({
            "benchmark": "long_context", "model_label": label,
            "id": s["id"], "prompt": s["prompt"][:100],
            "expected_min_tokens": min_tokens, "met_length": met,
            "response_preview": text[:200],
            **perf_fields(resp),
        })
        mark = "✓" if met else "short"
        print(f"      {mark} {resp.generation_tokens} tokens | decode {resp.generation_tps:.1f} tok/s | peak {resp.peak_memory:.2f} GB")

    met_rate = sum(r["met_length"] for r in results) / len(results) * 100
    print(f"  Length requirement met: {met_rate:.0f}%")
    return results, {"length_met_rate": round(met_rate, 1), "n": len(results)}


# ── Context scaling ───────────────────────────────────────────────────────────

def run_context_scaling(model, tokenizer):
    print(f"\n[Context scaling] {len(CONTEXT_LENGTHS)} lengths")
    base = "The quick brown fox jumps over the lazy dog. " * 20
    results = []

    for i, target in enumerate(CONTEXT_LENGTHS):
        prompt = (base * ((target // len(base)) + 1))[:target * 4]
        print(f"  [{i+1}/{len(CONTEXT_LENGTHS)}] ~{target} tokens context...")
        mx.clear_cache()
        try:
            _, resp = run(model, tokenizer, prompt, max_tokens=100)
            results.append({
                "target_tokens": target,
                "prompt_tokens": resp.prompt_tokens,
                "prompt_tps": round(resp.prompt_tps, 2),
                "generation_tps": round(resp.generation_tps, 2),
                "peak_memory_gb": round(resp.peak_memory, 3),
                "status": "ok",
            })
            print(f"      prefill {resp.prompt_tps:.1f} tok/s | decode {resp.generation_tps:.1f} tok/s | peak {resp.peak_memory:.2f} GB")
        except Exception as e:
            results.append({"target_tokens": target, "status": f"error: {str(e)[:80]}"})
            print(f"      ERROR: {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    ("gsm8k",        run_gsm8k),
    ("humaneval",    run_humaneval),
    ("mmlu",         run_mmlu),
    ("long_context", run_long_context),
]


def run_benchmark(model_path, label=None):
    if label is None:
        label = Path(model_path).name if Path(model_path).exists() else model_path.replace("/", "_")

    out_dir = OUTPUTS_DIR / (Path(model_path).name if Path(model_path).exists() else label)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model : {label}")
    print(f"Path  : {model_path}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    print("\nLoading model...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    print(f"Model loaded. Active Metal memory: {mx.get_active_memory() / 1024**3:.2f} GB\n")

    summary = {"label": label, "model_path": str(model_path), "benchmarks": {}}

    for name, fn in BENCHMARKS:
        results, stats = fn(model, tokenizer, label)
        summary["benchmarks"][name] = stats

        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {len(results)} records → {out_path}")

    # context scaling (no accuracy, just perf curve)
    ctx_results = run_context_scaling(model, tokenizer)
    summary["context_scaling"] = ctx_results
    with open(out_dir / "context_scaling.json", "w") as f:
        json.dump(ctx_results, f, indent=2)

    # overall perf summary across all samples
    all_records = []
    for name, _ in BENCHMARKS:
        p = out_dir / f"{name}.jsonl"
        if p.exists():
            all_records.extend(load_jsonl(p))

    if all_records:
        avg_decode = sum(r["generation_tps"] for r in all_records) / len(all_records)
        avg_prefill = sum(r["prompt_tps"] for r in all_records) / len(all_records)
        peak_mem = max(r["peak_memory_gb"] for r in all_records)
        summary["perf"] = {
            "avg_prefill_tps": round(avg_prefill, 2),
            "avg_decode_tps": round(avg_decode, 2),
            "peak_memory_gb": round(peak_mem, 3),
            "total_samples": len(all_records),
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY — {label}")
    print(f"{'='*60}")
    b = summary["benchmarks"]
    print(f"  GSM8K accuracy   : {b['gsm8k'].get('accuracy', '?')}%")
    print(f"  MMLU accuracy    : {b['mmlu'].get('accuracy', '?')}%")
    print(f"  HumanEval pass@1 : {b['humaneval'].get('pass_at_1', '?')}%")
    if "perf" in summary:
        p = summary["perf"]
        print(f"  Avg prefill      : {p['avg_prefill_tps']} tok/s")
        print(f"  Avg decode       : {p['avg_decode_tps']} tok/s")
        print(f"  Peak memory      : {p['peak_memory_gb']} GB")
    print(f"\nAll results saved → {out_dir}")

    return summary


def discover_models():
    models_dir = Path(__file__).parent.parent / "models"
    return sorted(models_dir.iterdir()) if models_dir.exists() else []


def print_comparison(summaries):
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'GSM8K':>8} {'MMLU':>8} {'HE pass@1':>10} {'Decode tok/s':>14} {'Peak GB':>8}")
    print("-" * 70)
    for s in summaries:
        b = s.get("benchmarks", {})
        p = s.get("perf", {})
        print(
            f"{s['label']:<20}"
            f"{str(b.get('gsm8k', {}).get('accuracy', '?'))+'%':>8}"
            f"{str(b.get('mmlu', {}).get('accuracy', '?'))+'%':>8}"
            f"{str(b.get('humaneval', {}).get('pass_at_1', '?'))+'%':>10}"
            f"{str(p.get('avg_decode_tps', '?')):>14}"
            f"{str(p.get('peak_memory_gb', '?')):>8}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model path or HF repo")
    parser.add_argument("--label", type=str, help="Label for output files")
    parser.add_argument("--all", action="store_true", help="Benchmark all models in ./models/")
    args = parser.parse_args()

    summaries = []

    if args.all:
        models = discover_models()
        if not models:
            print("No models found in ./models/")
        for i, m in enumerate(models):
            summaries.append(run_benchmark(str(m)))
            if i < len(models) - 1:
                print(f"\nCooling down for 2 minutes before next model...")
                for remaining in range(120, 0, -10):
                    print(f"  {remaining}s remaining...", end="\r", flush=True)
                    time.sleep(10)
                print(f"  Done. Starting next model.        ")
    elif args.model:
        summaries.append(run_benchmark(args.model, args.label))
    else:
        parser.print_help()

    if len(summaries) > 1:
        print_comparison(summaries)
