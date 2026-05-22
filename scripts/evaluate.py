"""
Quality evaluation pipeline: GSM8K, HumanEval, MMLU, long-context, manual.

Usage:
  python scripts/evaluate.py --model ./models/granite-4.1-8b-4bit --label 4bit
  python scripts/evaluate.py --model ibm-granite/granite-4.1-8b --label fp16
  python scripts/evaluate.py --all
"""

import argparse
import json
import re
import time
from pathlib import Path

import psutil
from mlx_lm import load, generate

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_generation(model, tokenizer, prompt, max_tokens=512):
    mem_before = get_rss_mb()
    t_start = time.perf_counter()
    ttft = None
    tokens = []

    for token in generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False):
        if ttft is None:
            ttft = (time.perf_counter() - t_start) * 1000
        tokens.append(token)

    total_s = time.perf_counter() - t_start
    mem_peak = get_rss_mb()

    response = tokenizer.decode(tokens) if tokens else ""
    tps = len(tokens) / total_s if total_s > 0 else 0

    return {
        "response": response,
        "tokens_generated": len(tokens),
        "ttft_ms": round(ttft or 0, 2),
        "total_time_s": round(total_s, 3),
        "tokens_per_sec": round(tps, 2),
        "mem_peak_mb": round(mem_peak, 1),
    }


# ── GSM8K ─────────────────────────────────────────────────────────────────────

GSM8K_PROMPT = """\
Solve the following math problem step by step. At the end, write your final answer as:
ANSWER: <number>

Problem: {question}"""


def extract_gsm8k_answer(response):
    match = re.search(r"ANSWER:\s*([\d,\.]+)", response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+\.?\d*", response)
    return nums[-1].replace(",", "") if nums else None


def extract_gsm8k_gold(answer_text):
    match = re.search(r"####\s*([\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip().split("\n")[-1].replace(",", "").strip()


def eval_gsm8k(model, tokenizer, label):
    path = DATASETS_DIR / "gsm8k.jsonl"
    if not path.exists():
        print("  GSM8K dataset not found — run setup_datasets.py first")
        return []

    samples = load_jsonl(path)
    print(f"  GSM8K: {len(samples)} samples")
    results = []

    for i, s in enumerate(samples):
        prompt = GSM8K_PROMPT.format(question=s["question"])
        gen = run_generation(model, tokenizer, prompt, max_tokens=300)

        predicted = extract_gsm8k_answer(gen["response"])
        gold = extract_gsm8k_gold(s["answer"])
        correct = predicted == gold

        record = {
            "benchmark": "gsm8k",
            "id": f"gsm8k_{i}",
            "model_label": label,
            "question": s["question"],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": correct,
            **gen,
        }
        results.append(record)

        status = "✓" if correct else "✗"
        print(f"    [{i+1}/{len(samples)}] {status} pred={predicted} gold={gold} | {gen['tokens_per_sec']:.1f} tok/s")

    acc = sum(r["correct"] for r in results) / len(results) * 100
    print(f"  GSM8K accuracy: {acc:.1f}%")
    return results


# ── HumanEval ─────────────────────────────────────────────────────────────────

HUMANEVAL_PROMPT = """\
Complete the following Python function. Write only the function body (the implementation), no explanation.

{prompt}
"""


def eval_humaneval(model, tokenizer, label):
    path = DATASETS_DIR / "humaneval.jsonl"
    if not path.exists():
        print("  HumanEval dataset not found — run setup_datasets.py first")
        return []

    samples = load_jsonl(path)
    print(f"  HumanEval: {len(samples)} samples")
    results = []

    for i, s in enumerate(samples):
        prompt = HUMANEVAL_PROMPT.format(prompt=s["prompt"])
        gen = run_generation(model, tokenizer, prompt, max_tokens=400)

        # Extract code block if model wraps in markdown
        code = gen["response"]
        md_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
        if md_match:
            code = md_match.group(1)

        # Syntactically valid?
        try:
            compile(s["prompt"] + "\n" + code, "<string>", "exec")
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        # Try to run tests
        passed = False
        error_msg = ""
        if syntax_ok:
            try:
                full_code = s["prompt"] + "\n" + code + "\n" + s["test"] + f"\ncheck({s['entry_point']})"
                exec(compile(full_code, "<string>", "exec"), {})
                passed = True
            except Exception as e:
                error_msg = str(e)[:100]

        record = {
            "benchmark": "humaneval",
            "id": s["task_id"],
            "model_label": label,
            "entry_point": s["entry_point"],
            "syntax_ok": syntax_ok,
            "tests_passed": passed,
            "error": error_msg,
            "generated_code": code[:500],
            **gen,
        }
        results.append(record)

        status = "✓" if passed else ("~" if syntax_ok else "✗")
        print(f"    [{i+1}/{len(samples)}] {status} {s['task_id']} | {gen['tokens_per_sec']:.1f} tok/s")

    pass_at_1 = sum(r["tests_passed"] for r in results) / len(results) * 100
    syntax_rate = sum(r["syntax_ok"] for r in results) / len(results) * 100
    print(f"  HumanEval pass@1: {pass_at_1:.1f}% | syntax ok: {syntax_rate:.1f}%")
    return results


# ── MMLU ──────────────────────────────────────────────────────────────────────

MMLU_PROMPT = """\
Answer the following multiple choice question. Reply with only the letter A, B, C, or D.

Question: {question}
A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""

MMLU_CHOICES = ["A", "B", "C", "D"]


def eval_mmlu(model, tokenizer, label):
    path = DATASETS_DIR / "mmlu.jsonl"
    if not path.exists():
        print("  MMLU dataset not found — run setup_datasets.py first")
        return []

    samples = load_jsonl(path)
    print(f"  MMLU: {len(samples)} samples")
    results = []

    for i, s in enumerate(samples):
        choices = s["choices"]
        while len(choices) < 4:
            choices.append("")
        prompt = MMLU_PROMPT.format(
            question=s["question"],
            a=choices[0], b=choices[1], c=choices[2], d=choices[3],
        )
        gen = run_generation(model, tokenizer, prompt, max_tokens=10)

        resp = gen["response"].strip().upper()
        predicted_letter = resp[0] if resp and resp[0] in MMLU_CHOICES else None
        gold_letter = MMLU_CHOICES[s["answer"]] if s["answer"] < 4 else None
        correct = predicted_letter == gold_letter

        record = {
            "benchmark": "mmlu",
            "id": f"mmlu_{i}",
            "model_label": label,
            "subject": s["subject"],
            "gold_answer": gold_letter,
            "predicted_answer": predicted_letter,
            "raw_response": gen["response"][:30],
            "correct": correct,
            **gen,
        }
        results.append(record)

        status = "✓" if correct else "✗"
        print(f"    [{i+1}/{len(samples)}] {status} pred={predicted_letter} gold={gold_letter} [{s['subject']}]")

    acc = sum(r["correct"] for r in results) / len(results) * 100
    print(f"  MMLU accuracy: {acc:.1f}%")
    return results


# ── Long-context ──────────────────────────────────────────────────────────────

def eval_long_context(model, tokenizer, label):
    path = DATASETS_DIR / "long_context_prompts.jsonl"
    if not path.exists():
        print("  Long-context prompts not found — run setup_datasets.py first")
        return []

    samples = load_jsonl(path)
    print(f"  Long-context: {len(samples)} samples")
    results = []

    for i, s in enumerate(samples):
        gen = run_generation(model, tokenizer, s["prompt"], max_tokens=600)
        min_tokens = s.get("expected_min_tokens", 300)
        met_length = gen["tokens_generated"] >= min_tokens

        record = {
            "benchmark": "long_context",
            "id": s["id"],
            "model_label": label,
            "prompt": s["prompt"][:100] + "...",
            "expected_min_tokens": min_tokens,
            "met_length_requirement": met_length,
            "response_preview": gen["response"][:200],
            **gen,
        }
        results.append(record)

        status = "✓" if met_length else "short"
        print(f"    [{i+1}/{len(samples)}] {status} {gen['tokens_generated']} tokens | {gen['tokens_per_sec']:.1f} tok/s")

    return results


# ── Manual prompts ────────────────────────────────────────────────────────────

def eval_manual(model, tokenizer, label):
    path = DATASETS_DIR / "manual_prompts.jsonl"
    if not path.exists():
        print("  Manual prompts not found — run setup_datasets.py first")
        return []

    samples = load_jsonl(path)
    print(f"  Manual: {len(samples)} samples")
    results = []

    for i, s in enumerate(samples):
        gen = run_generation(model, tokenizer, s["prompt"], max_tokens=400)

        record = {
            "benchmark": "manual",
            "id": s["id"],
            "model_label": label,
            "category": s["category"],
            "prompt": s["prompt"],
            "response": gen["response"],
            **gen,
        }
        results.append(record)
        print(f"    [{i+1}/{len(samples)}] [{s['category']}] {gen['tokens_per_sec']:.1f} tok/s")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(model_path, label=None):
    if label is None:
        label = Path(model_path).name if Path(model_path).exists() else model_path.replace("/", "_")

    out_dir = OUTPUTS_DIR / label
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    print(f"\nLoading model...")
    model, tokenizer = load(model_path)
    print(f"Model loaded.")

    all_results = []
    benchmarks = [
        ("GSM8K", eval_gsm8k),
        ("HumanEval", eval_humaneval),
        ("MMLU", eval_mmlu),
        ("Long-context", eval_long_context),
        ("Manual", eval_manual),
    ]

    summary = {"label": label, "model_path": str(model_path)}

    for name, fn in benchmarks:
        print(f"\n[{name}]")
        results = fn(model, tokenizer, label)
        all_results.extend(results)

        out_path = out_dir / f"{name.lower().replace('-','_')}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  → Saved {len(results)} records to {out_path}")

        # compute summary stats per benchmark
        if results:
            if name == "GSM8K":
                summary["gsm8k_acc"] = round(sum(r["correct"] for r in results) / len(results) * 100, 1)
                summary["gsm8k_n"] = len(results)
            elif name == "HumanEval":
                summary["humaneval_pass1"] = round(sum(r["tests_passed"] for r in results) / len(results) * 100, 1)
                summary["humaneval_n"] = len(results)
            elif name == "MMLU":
                summary["mmlu_acc"] = round(sum(r["correct"] for r in results) / len(results) * 100, 1)
                summary["mmlu_n"] = len(results)
            avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
            summary[f"{name.lower().replace('-','_')}_avg_tps"] = round(avg_tps, 2)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    return summary


def discover_models():
    models_dir = Path(__file__).parent.parent / "models"
    return sorted(models_dir.iterdir()) if models_dir.exists() else []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model path or HF repo")
    parser.add_argument("--label", type=str, help="Label for output files")
    parser.add_argument("--all", action="store_true", help="Evaluate all models in ./models/")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["gsm8k", "humaneval", "mmlu", "long_context", "manual"],
        help="Run only specific benchmarks",
    )
    args = parser.parse_args()

    summaries = []

    if args.all:
        for m in discover_models():
            summaries.append(run_evaluation(str(m)))
    elif args.model:
        summaries.append(run_evaluation(args.model, args.label))
    else:
        parser.print_help()

    if len(summaries) > 1:
        print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
        for s in summaries:
            print(f"{s['label']}: GSM8K={s.get('gsm8k_acc','?')}% | MMLU={s.get('mmlu_acc','?')}% | HumanEval={s.get('humaneval_pass1','?')}%")
