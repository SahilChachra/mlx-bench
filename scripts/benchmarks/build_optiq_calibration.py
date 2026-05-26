"""
Build a calibration JSONL for mlx-optiq matching its documented schema:
  {"domain": "prose",   "text": "..."}              # raw text
  {"domain": "thought", "messages": [...]}          # multi-turn chat
  {"domain": "code",    "messages": [...]}          # chat with code
  {"domain": "constraint", "messages": [...]}       # instruction-following

The PyPI v0.1.0 package ships without the bundled optiq.jsonl, so we rebuild
it from real public datasets covering the same domains.

Total: ~32 samples across 4 domains.

Output: datasets/optiq_calibration.jsonl
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parents[2] / "datasets" / "optiq_calibration.jsonl"
OUT.parent.mkdir(exist_ok=True)

rng = random.Random(0)
records = []


# ── prose (8) ─────────────────────────────────────────────────────────────────
print("[prose] loading wikitext-2…")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
prose_pool = [r["text"].strip() for r in ds if len(r["text"].strip()) > 800]
rng.shuffle(prose_pool)
for t in prose_pool[:8]:
    records.append({"domain": "prose", "text": t[:4000]})
print(f"  wrote {sum(r['domain'] == 'prose' for r in records)} prose")


# ── thought (8) ───────────────────────────────────────────────────────────────
print("[thought] loading MATH-500…")
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
math_pool = list(ds)
rng.shuffle(math_pool)
for s in math_pool[:8]:
    records.append({
        "domain": "thought",
        "messages": [
            {"role": "user", "content": f"Solve this problem step by step.\n\n{s['problem']}"},
            {"role": "assistant", "content": s["solution"][:2000]},
        ],
    })
print(f"  wrote 8 thought")


# ── code (8) ──────────────────────────────────────────────────────────────────
print("[code] loading HumanEval…")
ds = load_dataset("openai/openai_humaneval", split="test")
code_pool = list(ds)
rng.shuffle(code_pool)
for s in code_pool[:8]:
    records.append({
        "domain": "code",
        "messages": [
            {"role": "user", "content": f"Complete this Python function:\n\n```python\n{s['prompt']}\n```"},
            {"role": "assistant", "content": f"```python\n{s['prompt']}{s['canonical_solution']}\n```"},
        ],
    })
print(f"  wrote 8 code")


# ── constraint (8) ────────────────────────────────────────────────────────────
print("[constraint] loading IFEval…")
ds = load_dataset("google/IFEval", split="train")
if_pool = list(ds)
rng.shuffle(if_pool)
for s in if_pool[:8]:
    records.append({
        "domain": "constraint",
        "messages": [
            {"role": "user", "content": s["prompt"]},
            {"role": "assistant", "content": "OK, here is the response that follows your instructions."},
        ],
    })
print(f"  wrote 8 constraint")


# ── write ─────────────────────────────────────────────────────────────────────
with OUT.open("w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"\nTotal: {len(records)} samples → {OUT}")
domains = {}
for r in records:
    domains[r["domain"]] = domains.get(r["domain"], 0) + 1
print(f"Domains: {domains}")
