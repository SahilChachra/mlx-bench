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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.loader import load_model as load, text_stream_generate as stream_generate

_REPO = Path(__file__).resolve().parents[2]
DATASETS_DIR = _REPO / "datasets"
OUTPUTS_DIR  = _REPO / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CONTEXT_LENGTHS = [128, 256, 512, 1024]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_chat(tokenizer, user_text, system=None, enable_thinking=True):
    """Wrap user_text in the tokenizer's chat template when available.

    Falls back to raw text for base models. Tries enable_thinking=True for
    hybrid-reasoning templates (MiniCPM5, Qwen3, etc.); silently drops the kwarg
    if the template doesn't accept it.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return user_text
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text):
    """Remove thinking content so answer extraction sees the final reply.

    Handles two cases:
      1. Full <think>...</think> block in the output.
      2. Output that starts with thinking content and only contains the closing
         </think> tag (happens when the chat template injects <think> into the
         prompt itself, e.g. MiniCPM5).
    """
    text = THINK_RE.sub("", text)
    if "</think>" in text.lower():
        text = THINK_CLOSE_RE.sub("", text)
    return text


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
        user_msg = GSM8K_PROMPT.format(question=s["question"])
        prompt = format_chat(tokenizer, user_msg)
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=4096)

        answer_text = strip_thinking(text)
        predicted = extract_gsm8k_answer(answer_text)
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
HUMANEVAL_INSTRUCT = (
    "Complete the following Python function. Return only the complete function "
    "(including the signature and body) inside a ```python code block.\n\n"
    "```python\n{prompt}```"
)


def _dedent_block(text):
    """Strip the minimum common indentation from all non-empty lines."""
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    return "\n".join(l[indent:] if len(l) >= indent else l for l in lines)


def _extract_code_chat(response):
    """Extract a full top-level function from a chat-mode response.

    Looks inside the LAST ```python``` block (skipping any thinking-block echoes
    above), strips minimum common indentation, and returns the dedented text.
    Fallback: return the response after stripping fences.
    """
    blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if blocks:
        return _dedent_block(blocks[-1])
    cleaned = re.sub(r"```\w*\s*", "", response).strip()
    return _dedent_block(cleaned)


def _extract_code_completion(prompt, response):
    """Extract a function body when the model is completing a base-mode prompt."""
    if response.startswith(prompt.strip()):
        response = response[len(prompt.strip()):]
    md = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if md:
        text = md.group(1)
    else:
        response = re.sub(r"```\w*\s*$", "", response.rstrip(), flags=re.MULTILINE).rstrip()
        cut = re.search(r"\n(?=def |class |if __name__)", response)
        if cut:
            response = response[:cut.start()]
        text = response
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

    has_chat = getattr(tokenizer, "chat_template", None) is not None
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['task_id']} — {s['entry_point']}")
        mx.clear_cache()
        if has_chat:
            prompt = format_chat(tokenizer, HUMANEVAL_INSTRUCT.format(prompt=s["prompt"]))
        else:
            prompt = HUMANEVAL_PROMPT.format(prompt=s["prompt"])
        text, resp = run(model, tokenizer, prompt, max_tokens=4096)

        post = strip_thinking(text)
        if has_chat:
            # Chat mode returns the full function definition. Use it as-is.
            code = _extract_code_chat(post)
            source = code
        else:
            # Base mode returns just the function body to be appended to prompt.
            code = _extract_code_completion(s["prompt"], post)
            source = s["prompt"] + "\n" + code

        try:
            compile(source, "<string>", "exec")
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        passed, error_msg = False, ""
        if syntax_ok:
            try:
                exec(compile(source + "\n" + s["test"] + f"\ncheck({s['entry_point']})", "<string>", "exec"), {})
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
        user_msg = MMLU_PROMPT.format(
            question=s["question"], a=choices[0], b=choices[1], c=choices[2], d=choices[3]
        )
        has_chat = getattr(tokenizer, "chat_template", None) is not None
        prompt = format_chat(tokenizer, user_msg) if has_chat else user_msg
        # Chat / thinking models need room to think before emitting a letter.
        # Base models with no chat template answer in 1 token; cap tight to keep it fast.
        mt = 1024 if has_chat else 10
        print(f"  [{i+1}/{len(samples)}] [{s['subject']}] {s['question'][:70]}...")
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=mt)

        answer_text = strip_thinking(text).strip().upper()
        m = re.search(r"\b([ABCD])\b", answer_text)
        predicted = m.group(1) if m else (answer_text[0] if answer_text and answer_text[0] in CHOICES else None)
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


# ── AIME 2024 ─────────────────────────────────────────────────────────────────

AIME_PROMPT = """\
Solve the following math problem. The answer is a non-negative integer between 0 and 999. \
Place your final answer inside \\boxed{{}}.

Problem: {problem}"""


def _extract_boxed(text):
    """Return the contents of the LAST \\boxed{...}, handling nested braces."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth = 1
    out = []
    while j < len(text) and depth > 0:
        c = text[j]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        j += 1
    return "".join(out).strip() if depth == 0 else None


def _extract_int_answer(text):
    boxed = _extract_boxed(text)
    src = boxed if boxed else text
    m = re.findall(r"-?\d+", src.replace(",", ""))
    return m[-1] if m else None


def run_aime(model, tokenizer, label):
    path = DATASETS_DIR / "aime_2024.jsonl"
    samples = load_jsonl(path)
    print(f"\n[AIME 2024] {len(samples)} problems — competition math, integer answer")
    results = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] AIME-{s['id']}: {s['problem'][:80]}...")
        user_msg = AIME_PROMPT.format(problem=s["problem"])
        prompt = format_chat(tokenizer, user_msg)
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=16384)

        answer_text = strip_thinking(text)
        predicted = _extract_int_answer(answer_text)
        gold = s["answer"].strip()
        correct = predicted == gold

        results.append({
            "benchmark": "aime_2024", "model_label": label,
            "id": s["id"], "problem": s["problem"][:200],
            "gold": gold, "predicted": predicted, "correct": correct,
            "response_tail": text[-300:],
            **perf_fields(resp),
        })
        mark = "✓" if correct else "✗"
        print(f"      {mark} pred={predicted} gold={gold} | {resp.generation_tokens} tok | decode {resp.generation_tps:.1f} tok/s")

    acc = sum(r["correct"] for r in results) / len(results) * 100
    print(f"  AIME 2024 accuracy: {acc:.1f}%  ({sum(r['correct'] for r in results)}/{len(results)})")
    return results, {"accuracy": round(acc, 1), "n": len(results)}


# ── MATH-500 ──────────────────────────────────────────────────────────────────

MATH500_PROMPT = """\
Solve the following math problem. Place your final answer inside \\boxed{{}}.

Problem: {problem}"""


def _normalize_math(s):
    if s is None:
        return None
    s = s.strip()
    # Drop common LaTeX wrappers and whitespace
    s = s.replace(" ", "").replace("\n", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("$", "")
    # Trim outer parens/brackets if they wrap the whole thing
    while len(s) >= 2 and ((s[0] == "(" and s[-1] == ")") or (s[0] == "[" and s[-1] == "]")):
        s = s[1:-1]
    return s.lower()


def run_math500(model, tokenizer, label):
    """Three-state scoring: correct / wrong / no_answer.

    no_answer = model never closed </think> (when chat template injects <think>)
    or never produced a \\boxed{} in the post-thinking text.
    """
    path = DATASETS_DIR / "math500.jsonl"
    samples = load_jsonl(path)
    print(f"\n[MATH-500] {len(samples)} problems — math reasoning (boxed answer)")
    results = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] L{s['level']} [{s['subject']}]: {s['problem'][:80]}...")
        user_msg = MATH500_PROMPT.format(problem=s["problem"])
        prompt = format_chat(tokenizer, user_msg)
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=8192)

        finished = "</think>" in text.lower()
        answer_text = strip_thinking(text) if finished else ""
        predicted = _extract_boxed(answer_text) if answer_text else None

        if predicted is None:
            status = "no_answer"
        elif _normalize_math(predicted) == _normalize_math(s["answer"]):
            status = "correct"
        else:
            status = "wrong"

        results.append({
            "benchmark": "math500", "model_label": label,
            "id": s["id"], "level": s["level"], "subject": s["subject"],
            "problem": s["problem"][:200],
            "gold": s["answer"], "predicted": predicted,
            "finished_thinking": finished, "status": status,
            "correct": status == "correct",
            **perf_fields(resp),
        })
        mark = {"correct": "✓", "wrong": "✗", "no_answer": "—"}[status]
        print(f"      {mark} pred={(predicted or '<none>')[:40]} gold={s['answer'][:40]} | {resp.generation_tokens} tok | decode {resp.generation_tps:.1f} tok/s")

    n = len(results)
    n_correct = sum(r["status"] == "correct" for r in results)
    n_answered = sum(r["status"] in ("correct", "wrong") for r in results)
    acc = n_correct / n * 100
    answer_rate = n_answered / n * 100
    # per-level breakdown
    by_lvl = {}
    for r in results:
        d = by_lvl.setdefault(r["level"], {"n": 0, "c": 0})
        d["n"] += 1
        d["c"] += int(r["status"] == "correct")
    per_level = {f"level_{k}": round(v["c"] / v["n"] * 100, 1) for k, v in sorted(by_lvl.items())}
    print(f"  MATH-500 accuracy: {acc:.1f}%  (answered {n_answered}/{n} = {answer_rate:.0f}%)  per-level: {per_level}")
    return results, {
        "accuracy": round(acc, 1),
        "n": n,
        "n_correct": n_correct,
        "n_answered": n_answered,
        "answer_rate": round(answer_rate, 1),
        "per_level": per_level,
    }


# ── IFEval (single-rule subset) ───────────────────────────────────────────────

def _count_sentences(text):
    # Crude: split on .!? followed by whitespace/end. Good enough for IFEval scoring.
    sents = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return sum(1 for s in sents if s.strip())


def _count_paragraphs(text):
    paras = re.split(r"\n\s*\n", text.strip())
    return sum(1 for p in paras if p.strip())


def _check_relation(actual, relation, target):
    """IFEval length-constraint relations."""
    if target is None:
        return None
    target = int(target)
    relation = (relation or "").lower()
    if relation in ("at least", "more than"):
        return actual > target if "more" in relation else actual >= target
    if relation in ("at most", "less than"):
        return actual < target if "less" in relation else actual <= target
    if relation in ("exactly", ""):
        return actual == target
    return None


def _score_ifeval(rule, text, kwargs):
    """Return True/False/None (None = unsupported)."""
    t = text or ""

    if rule == "punctuation:no_comma":
        return "," not in t

    if rule == "length_constraints:number_words":
        n = len(t.split())
        return _check_relation(n, kwargs.get("relation"), kwargs.get("num_words"))

    if rule == "length_constraints:number_sentences":
        n = _count_sentences(t)
        return _check_relation(n, kwargs.get("relation"), kwargs.get("num_sentences"))

    if rule == "length_constraints:number_paragraphs":
        n = _count_paragraphs(t)
        return _check_relation(n, kwargs.get("relation"), kwargs.get("num_paragraphs"))

    if rule == "change_case:english_lowercase":
        letters = [c for c in t if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)

    if rule == "change_case:english_capital":
        letters = [c for c in t if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    if rule == "keywords:forbidden_words":
        words = kwargs.get("forbidden_words") or []
        lower = t.lower()
        return all(w.lower() not in lower for w in words)

    if rule == "keywords:existence":
        words = kwargs.get("keywords") or []
        lower = t.lower()
        return all(w.lower() in lower for w in words)

    if rule == "detectable_format:number_highlighted_sections":
        # Asterisk-delimited highlights: *foo*, but not ** (bold) or *** (header).
        matches = re.findall(r"(?<!\*)\*([^*\n][^*\n]*?)\*(?!\*)", t)
        return len(matches) >= int(kwargs.get("num_highlights", 0))

    if rule == "detectable_format:number_bullet_lists":
        # Bullets: lines starting with - or * followed by space
        bullets = [ln for ln in t.splitlines() if re.match(r"^\s*[-*]\s+", ln)]
        target = kwargs.get("num_bullets")
        if target is None:
            return None
        return len(bullets) == int(target)

    if rule == "detectable_content:number_placeholders":
        placeholders = re.findall(r"\[[^\[\]\n]+\]", t)
        return len(placeholders) >= int(kwargs.get("num_placeholders", 0))

    return None


def run_ifeval(model, tokenizer, label):
    path = DATASETS_DIR / "ifeval.jsonl"
    samples = load_jsonl(path)
    print(f"\n[IFEval] {len(samples)} prompts — instruction following (single-rule subset)")
    results = []

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['rule']}: {s['prompt'][:70]}...")
        # IFEval is instruction following, not reasoning. Disable thinking so the
        # model writes the response directly — same as the official IFEval setup.
        prompt = format_chat(tokenizer, s["prompt"], enable_thinking=False)
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=2048)

        response = strip_thinking(text).strip()
        passed = _score_ifeval(s["rule"], response, s.get("kwargs", {}))
        status = "correct" if passed else ("wrong" if passed is False else "unscored")

        results.append({
            "benchmark": "ifeval", "model_label": label,
            "id": s["id"], "rule": s["rule"],
            "prompt": s["prompt"][:200],
            "response_preview": response[:300],
            "status": status, "correct": status == "correct",
            **perf_fields(resp),
        })
        mark = {"correct": "✓", "wrong": "✗", "unscored": "?"}[status]
        print(f"      {mark} {s['rule']} | {resp.generation_tokens} tok | decode {resp.generation_tps:.1f} tok/s")

    n = len(results)
    n_scored = sum(r["status"] in ("correct", "wrong") for r in results)
    n_correct = sum(r["status"] == "correct" for r in results)
    acc = n_correct / max(n_scored, 1) * 100
    # per-rule-family accuracy
    by_family = {}
    for r in results:
        fam = r["rule"].split(":")[0]
        d = by_family.setdefault(fam, {"n": 0, "c": 0})
        if r["status"] in ("correct", "wrong"):
            d["n"] += 1
            d["c"] += int(r["status"] == "correct")
    per_family = {k: round(v["c"] / v["n"] * 100, 1) for k, v in sorted(by_family.items()) if v["n"]}
    print(f"  IFEval accuracy: {acc:.1f}%  (scored {n_scored}/{n})  per-family: {per_family}")
    return results, {
        "accuracy": round(acc, 1),
        "n": n,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "per_family": per_family,
    }


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
    ("math500",      run_math500),
    ("humaneval",    run_humaneval),
    ("ifeval",       run_ifeval),
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
    # Force MLX to drop any cached memory from previous models, then reset
    # the peak counter so it tracks only this model's footprint.
    mx.clear_cache()
    mx.reset_peak_memory()
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
    if "math500" in b:
        m = b["math500"]
        print(f"  MATH-500         : {m.get('accuracy', '?')}%  (answered {m.get('n_answered', '?')}/{m.get('n', '?')})")
    if "humaneval" in b:
        print(f"  HumanEval pass@1 : {b['humaneval'].get('pass_at_1', '?')}%")
    if "ifeval" in b:
        e = b["ifeval"]
        print(f"  IFEval           : {e.get('accuracy', '?')}%  (scored {e.get('n_scored', '?')}/{e.get('n', '?')})")
    if "gsm8k" in b:
        print(f"  GSM8K accuracy   : {b['gsm8k'].get('accuracy', '?')}%")
    if "mmlu" in b:
        print(f"  MMLU accuracy    : {b['mmlu'].get('accuracy', '?')}%")
    if "perf" in summary:
        p = summary["perf"]
        print(f"  Avg prefill      : {p['avg_prefill_tps']} tok/s")
        print(f"  Avg decode       : {p['avg_decode_tps']} tok/s")
        print(f"  Peak memory      : {p['peak_memory_gb']} GB")
    print(f"\nAll results saved → {out_dir}")

    return summary


def discover_models():
    models_dir = _REPO / "models"
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
