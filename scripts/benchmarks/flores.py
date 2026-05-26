"""
Translation benchmark for Hy-MT2 (or any translation-capable LLM).

Runs FLORES-200 devtest on a configurable set of language pairs, measures
chrF++ and BLEU via sacrebleu, and also captures prefill/decode tok/s and
peak memory from mlx-lm's GenerationResponse so a single pass covers both
quality and performance.

Usage:
  python scripts/flores_benchmark.py --model ./models/hy-mt2-7b-fp16 --label fp16
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import sacrebleu
from mlx_lm import load, stream_generate

_REPO = Path(__file__).resolve().parents[2]
DATASETS_DIR = _REPO / "datasets"
FLORES_DIR   = DATASETS_DIR / "flores200_dataset" / "devtest"
OUTPUTS_DIR  = _REPO / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

# (src_code, tgt_code, src_lang, tgt_lang)
DEFAULT_PAIRS = [
    ("eng_Latn", "fra_Latn", "English", "French"),
    ("eng_Latn", "deu_Latn", "English", "German"),
    ("eng_Latn", "zho_Hans", "English", "Chinese"),
    ("eng_Latn", "jpn_Jpan", "English", "Japanese"),
    ("eng_Latn", "spa_Latn", "English", "Spanish"),
    ("fra_Latn", "eng_Latn", "French",  "English"),
    ("zho_Hans", "eng_Latn", "Chinese", "English"),
    ("jpn_Jpan", "eng_Latn", "Japanese","English"),
]

CONTEXT_LENGTHS = [128, 256, 512, 1024]


def _read_lang(code):
    p = FLORES_DIR / f"{code}.devtest"
    if not p.exists():
        raise FileNotFoundError(
            f"FLORES file missing: {p}\n"
            f"Download with:\n"
            f"  cd datasets && curl -sLO https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz && tar -xzf flores200_dataset.tar.gz"
        )
    with open(p) as f:
        return [line.rstrip("\n") for line in f]


def ensure_flores(pairs, n_per_pair):
    """Build a JSONL cache from the local FLORES-200 devtest tarball."""
    path = DATASETS_DIR / f"flores_{n_per_pair}.jsonl"
    if path.exists():
        with open(path) as f:
            cached = [json.loads(line) for line in f if line.strip()]
        have = {(r["src_code"], r["tgt_code"]) for r in cached}
        if all((s, t) in have for s, t, _, _ in pairs):
            print(f"  Using cached FLORES → {path}")
            return path

    print(f"  Building FLORES sample for {len(pairs)} pairs × {n_per_pair} samples...")
    all_records = []
    cache = {}
    for src_code, tgt_code, src_lang, tgt_lang in pairs:
        if src_code not in cache:
            cache[src_code] = _read_lang(src_code)
        if tgt_code not in cache:
            cache[tgt_code] = _read_lang(tgt_code)
        for i in range(min(n_per_pair, len(cache[src_code]))):
            all_records.append({
                "src_code": src_code, "tgt_code": tgt_code,
                "src_lang": src_lang, "tgt_lang": tgt_lang,
                "src_text": cache[src_code][i],
                "ref_text": cache[tgt_code][i],
                "idx": i,
            })
    with open(path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_records)} samples → {path}")
    return path


def make_prompt(tokenizer, src_lang, tgt_lang, src_text):
    """Hy-MT2's recommended translation prompt format (English variant)."""
    user = (
        f"Translate the following text into {tgt_lang}. Note that you should "
        f"only output the translated result without any additional explanation:"
        f"\n\n{src_text}"
    )
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return user


def run(model, tokenizer, prompt, max_tokens=512):
    chunks, last, n = [], None, 0
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        chunks.append(resp.text)
        last = resp
        n += 1
    if last is None:
        raise RuntimeError("Model generated 0 tokens.")
    return "".join(chunks).strip(), last


def perf_fields(resp):
    return {
        "prompt_tokens":    resp.prompt_tokens,
        "generation_tokens": resp.generation_tokens,
        "prompt_tps":       round(resp.prompt_tps, 2),
        "generation_tps":   round(resp.generation_tps, 2),
        "peak_memory_gb":   round(resp.peak_memory, 3),
        "finish_reason":    resp.finish_reason,
    }


def clean_translation(text, tgt_lang):
    """Strip common boilerplate (markdown, role tags, leading 'X:')."""
    t = text.strip()
    # Drop leading "French:" / "Translation:" style labels if the model echoed one
    for prefix in (f"{tgt_lang}:", f"{tgt_lang.lower()}:", "Translation:", "translation:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    # Stop at the first empty line (model sometimes continues with new content)
    if "\n\n" in t:
        t = t.split("\n\n", 1)[0].strip()
    # Drop leading/trailing quotes if both present
    if len(t) > 2 and t[0] in '"“' and t[-1] in '"”':
        t = t[1:-1].strip()
    return t


def run_flores(model, tokenizer, label, flores_path):
    with open(flores_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"\n[FLORES-200] {len(samples)} samples — translation quality")

    results = []
    by_pair = {}

    for i, s in enumerate(samples):
        prompt = make_prompt(tokenizer, s["src_lang"], s["tgt_lang"], s["src_text"])
        print(f"  [{i+1}/{len(samples)}] {s['src_code']} → {s['tgt_code']}: {s['src_text'][:60]}...")
        mx.clear_cache()
        text, resp = run(model, tokenizer, prompt, max_tokens=256)

        hyp = clean_translation(text, s["tgt_lang"])

        record = {
            "benchmark": "flores", "model_label": label,
            "src_code": s["src_code"], "tgt_code": s["tgt_code"],
            "src_text": s["src_text"], "ref_text": s["ref_text"],
            "hypothesis": hyp,
            **perf_fields(resp),
        }
        results.append(record)
        pair_key = f"{s['src_code']}→{s['tgt_code']}"
        by_pair.setdefault(pair_key, []).append(record)
        print(f"      hyp: {hyp[:60]}...")
        print(f"      ref: {s['ref_text'][:60]}...")
        print(f"      decode {resp.generation_tps:.1f} tok/s | peak {resp.peak_memory:.2f} GB")

    # Per-pair chrF / BLEU
    per_pair = []
    all_hyps, all_refs = [], []
    for pair_key, recs in by_pair.items():
        hyps = [r["hypothesis"] for r in recs]
        refs = [r["ref_text"] for r in recs]
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        per_pair.append({
            "pair": pair_key,
            "n": len(hyps),
            "chrf": round(chrf, 2),
            "bleu": round(bleu, 2),
        })
        all_hyps.extend(hyps)
        all_refs.extend(refs)
        print(f"  {pair_key}: chrF++ {chrf:.2f}  BLEU {bleu:.2f}  (n={len(hyps)})")

    avg_chrf = sacrebleu.corpus_chrf(all_hyps, [all_refs], word_order=2).score
    avg_bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score

    summary = {
        "per_pair":  per_pair,
        "avg_chrf":  round(avg_chrf, 2),
        "avg_bleu":  round(avg_bleu, 2),
        "n":         len(results),
    }
    print(f"\n  Average chrF++ {avg_chrf:.2f}  BLEU {avg_bleu:.2f}  over {len(results)} samples")
    return results, summary


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


def run_benchmark(model_path, label=None, flores_path=None):
    if label is None:
        label = Path(model_path).name

    out_dir = OUTPUTS_DIR / Path(model_path).name
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model : {label}")
    print(f"Path  : {model_path}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    print("\nLoading model...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    mx.clear_cache()
    mx.reset_peak_memory()
    print(f"Model loaded. Active Metal memory: {mx.get_active_memory() / 1024**3:.2f} GB\n")

    summary = {"label": label, "model_path": str(model_path), "benchmarks": {}}

    results, stats = run_flores(model, tokenizer, label, flores_path)
    summary["benchmarks"]["flores"] = stats
    with open(out_dir / "flores.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(results)} records → {out_dir / 'flores.jsonl'}")

    ctx_results = run_context_scaling(model, tokenizer)
    summary["context_scaling"] = ctx_results
    with open(out_dir / "context_scaling.json", "w") as f:
        json.dump(ctx_results, f, indent=2)

    # overall perf summary across FLORES samples
    if results:
        avg_decode = sum(r["generation_tps"] for r in results) / len(results)
        avg_prefill = sum(r["prompt_tps"] for r in results) / len(results)
        peak_mem = max(r["peak_memory_gb"] for r in results)
        summary["perf"] = {
            "avg_prefill_tps": round(avg_prefill, 2),
            "avg_decode_tps": round(avg_decode, 2),
            "peak_memory_gb": round(peak_mem, 3),
            "total_samples": len(results),
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY — {label}")
    print(f"{'='*60}")
    fl = summary["benchmarks"]["flores"]
    print(f"  FLORES chrF++ : {fl['avg_chrf']}")
    print(f"  FLORES BLEU   : {fl['avg_bleu']}")
    if "perf" in summary:
        p = summary["perf"]
        print(f"  Avg prefill   : {p['avg_prefill_tps']} tok/s")
        print(f"  Avg decode    : {p['avg_decode_tps']} tok/s")
        print(f"  Peak memory   : {p['peak_memory_gb']} GB")
    print(f"\nAll results saved → {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF repo")
    parser.add_argument("--label", help="Label for output files")
    parser.add_argument("--n-per-pair", type=int, default=20,
                        help="Number of FLORES samples per language pair (default 20)")
    args = parser.parse_args()

    flores_path = ensure_flores(DEFAULT_PAIRS, args.n_per_pair)
    run_benchmark(args.model, args.label, flores_path)
