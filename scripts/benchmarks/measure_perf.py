"""
Steady-state decode-speed measurement.

Loads a model, warms up (so MLX kernel JIT is done), then runs N short
generations and reports the average. Patches the model's summary.json:
  - summary["perf"]["steady_state"] = {"decode_tps":..., "prompt_tps":..., "peak_gb":...}

Usage:
  python scripts/measure_perf.py --model models/minicpm5-1b-8bit --label 8bit
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

PROMPTS = [
    "Write a one-paragraph explanation of how transformers work.",
    "Summarize the plot of Hamlet in three sentences.",
    "Explain the difference between recursion and iteration with a short example.",
    "Describe the key ideas behind reinforcement learning.",
    "What are the main causes of climate change?",
]


def measure(path, label, max_tokens=200, repeats=3):
    print(f"Loading {label} from {path}…")
    model, tok = load(path)
    mx.eval(model.parameters())
    mx.clear_cache(); mx.reset_peak_memory()

    # Warmup — let MLX JIT and load kernels
    warm_prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    for _ in stream_generate(model, tok, prompt=warm_prompt, max_tokens=20):
        pass
    mx.clear_cache(); mx.reset_peak_memory()

    decode_tps_runs, prompt_tps_runs, peaks = [], [], []
    for i in range(repeats):
        p = PROMPTS[i % len(PROMPTS)]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        last = None
        for r in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens):
            last = r
        decode_tps_runs.append(last.generation_tps)
        prompt_tps_runs.append(last.prompt_tps)
        peaks.append(last.peak_memory)
        print(f"  [{i+1}/{repeats}] prompt_tps={last.prompt_tps:.1f}  decode_tps={last.generation_tps:.1f}  gen_tokens={last.generation_tokens}  peak={last.peak_memory:.2f} GB")

    return {
        "decode_tps": round(sum(decode_tps_runs) / len(decode_tps_runs), 2),
        "prompt_tps": round(sum(prompt_tps_runs) / len(prompt_tps_runs), 2),
        "peak_memory_gb": round(max(peaks), 3),
        "decode_tps_runs": [round(x, 2) for x in decode_tps_runs],
        "max_tokens": max_tokens,
        "repeats": repeats,
        "note": "Warmed, short-prompt, chat-templated, thinking disabled. Represents steady-state decode for typical chat use; long thinking traces will be slower due to KV-cache growth.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    stats = measure(args.model, args.label, args.max_tokens, args.repeats)

    # Patch summary
    summary_path = OUTPUTS_DIR / Path(args.model).name / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary.setdefault("perf", {})["steady_state"] = stats
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nPatched {summary_path}")
    print(f"\nSteady-state perf: {stats}")
