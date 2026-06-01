"""
RefCOCOg-val grounding benchmark for VLM quantization variants.

Loads N random samples from lmms-lab/RefCOCOg val, runs the VLM with a
referring-expression grounding prompt, parses the predicted `<box>…</box>`
coordinate tokens, and reports Acc@0.5, Acc@0.75, mean IoU, and parse-fail
rate against the ground-truth COCO bbox.

LocateAnything (and most Qwen2-VL-family models) emit bbox tokens like:
    <box><123><456><789><912></box>
or the older comma form:
    <box>(123,456),(789,912)</box>
both with coordinates normalized to 0..1000. We accept both forms.

Usage:
  MLX_BENCH_MODALITY=vlm \\
    python -m scripts.benchmarks.refcoco \\
      --model models/locateanything-3b-fp16 --label fp16 --n 200
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_NAME  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = _REPO / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Default prompt — NVIDIA's recommended phrasing for single-instance
# referring-expression grounding on LocateAnything. Override via --prompt for
# other VLMs (e.g. Qwen2.5-VL expects "Locate {phrase} in the image. Output
# the bounding box.").
DEFAULT_PROMPT = "<image-1>\nLocate a single instance that matches the following description: {phrase}."

# LocateAnything is designed for Parallel Box Decoding; mlx-vlm's default
# "slow" (pure AR) yields ~32% empty outputs on the bf16 base. Other VLMs
# may need a different mode — flag-controlled.
DEFAULT_GEN_MODE = "hybrid"
DEFAULT_MAX_TOKENS = 2048


def _register_locateanything():
    """Patch transformers to expose LocateAnything's custom processor +
    image-processor classes so ProcessorMixin's class-name lookup succeeds.
    Without this, mlx_vlm.load falls back to returning the bare tokenizer.
    No-op for other VLMs (the import simply fails).

    Only call this when running LocateAnything — gated by --register-locateanything
    so the script stays generic for other VLMs."""
    try:
        from mlx_vlm.models.locateanything.processing_locateanything import (
            LocateAnythingProcessor,
        )
        from mlx_vlm.models.locateanything.image_processing_locateanything import (
            LocateAnythingImageProcessor,
        )
        import transformers as _t
        _t.LocateAnythingProcessor = LocateAnythingProcessor
        _t.LocateAnythingImageProcessor = LocateAnythingImageProcessor
    except ImportError:
        pass  # not a locateanything checkout

# Two known coord-token formats — both 0..1000 normalized.
_BOX_RE_ANGLE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>")
_BOX_RE_PAREN = re.compile(r"<box>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</box>")


def parse_box(text):
    """Return (x1,y1,x2,y2) in 0..1000 if a box is present, else None."""
    for r in (_BOX_RE_ANGLE, _BOX_RE_PAREN):
        m = r.search(text)
        if m:
            return tuple(int(g) for g in m.groups())
    return None


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_refcoco(model, processor, samples, max_tokens=128, verbose=False,
                prompt_template=DEFAULT_PROMPT, generation_mode=DEFAULT_GEN_MODE):
    """Returns (per_sample_records, aggregates)."""
    from mlx_vlm import generate  # local import; only present in VLM env

    records = []
    n_hit_50 = n_hit_75 = n_parsed = 0
    sum_iou = 0.0
    prefill_tps = []
    decode_tps = []

    t_total = time.time()
    for i, s in enumerate(samples):
        img = s["image"]
        W, H = img.size
        phrase = s["answer"][0] if isinstance(s["answer"], list) else s["answer"]
        x, y, w, h = s["bbox"]
        gt = (x, y, x + w, y + h)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.convert("RGB").save(f.name, format="JPEG", quality=92)
            tmp_path = f.name

        try:
            prompt = prompt_template.format(phrase=phrase)
            t0 = time.time()
            resp = generate(model, processor, prompt=prompt, image=tmp_path,
                            max_tokens=max_tokens, generation_mode=generation_mode,
                            verbose=False)
            elapsed = time.time() - t0

            text = getattr(resp, "text", str(resp))
            ptps = getattr(resp, "prompt_tps", None)
            dtps = getattr(resp, "generation_tps", None)
            if ptps: prefill_tps.append(ptps)
            if dtps: decode_tps.append(dtps)

            box01k = parse_box(text)
            if box01k is None:
                rec = {"i": i, "phrase": phrase, "gt": gt, "pred_text": text[:200],
                       "iou": 0.0, "hit_50": False, "hit_75": False, "parse_ok": False,
                       "elapsed_s": round(elapsed, 2)}
            else:
                n_parsed += 1
                x1, y1, x2, y2 = box01k
                pred = (x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H)
                pred = (min(pred[0], pred[2]), min(pred[1], pred[3]),
                        max(pred[0], pred[2]), max(pred[1], pred[3]))
                v = iou(pred, gt)
                sum_iou += v
                hit_50 = v >= 0.5
                hit_75 = v >= 0.75
                n_hit_50 += int(hit_50)
                n_hit_75 += int(hit_75)
                rec = {"i": i, "phrase": phrase, "gt": gt, "pred_px": pred,
                       "pred_norm": box01k, "iou": round(v, 4),
                       "hit_50": hit_50, "hit_75": hit_75, "parse_ok": True,
                       "elapsed_s": round(elapsed, 2)}
            records.append(rec)
            if verbose or (i % 20 == 0):
                print(f"  [{i+1:>3}/{len(samples)}] iou={rec['iou']:.3f}"
                      f"  parse={'ok' if rec['parse_ok'] else 'FAIL'}"
                      f"  phrase='{phrase[:60]}'")
        finally:
            os.unlink(tmp_path)

    n = len(samples)
    aggs = {
        "n": n,
        "accuracy_50": round(100 * n_hit_50 / n, 2),
        "accuracy_75": round(100 * n_hit_75 / n, 2),
        "mean_iou": round(sum_iou / n, 4),
        "parse_fail_rate": round(100 * (n - n_parsed) / n, 2),
        "total_seconds": round(time.time() - t_total, 1),
    }
    if prefill_tps:
        aggs["avg_prefill_tps"] = round(sum(prefill_tps) / len(prefill_tps), 2)
    if decode_tps:
        aggs["avg_decode_tps"] = round(sum(decode_tps) / len(decode_tps), 2)
    return records, aggs


def load_samples(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/RefCOCOg", split="val")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n)
    return [ds[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to MLX model dir")
    ap.add_argument("--label", required=True, help="Variant label (e.g. fp16, mxfp4)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="Prompt template with {phrase} placeholder.")
    ap.add_argument("--generation-mode", default=DEFAULT_GEN_MODE,
                    choices=["slow", "hybrid", "fast"],
                    help="mlx-vlm decoding mode (hybrid = Parallel Box Decoding).")
    ap.add_argument("--register-locateanything", action="store_true",
                    help="Patch transformers with LocateAnything's processor classes "
                         "(workaround for the PR-branch registration bug).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.register_locateanything:
        _register_locateanything()
    from mlx_vlm import load as vlm_load
    print(f"Loading model: {args.model}")
    model, processor = vlm_load(args.model)

    print(f"Loading {args.n} samples from lmms-lab/RefCOCOg val (seed={args.seed})")
    samples = load_samples(args.n, args.seed)

    print(f"\nRunning RefCOCOg grounding on {args.label}...")
    records, aggs = run_refcoco(model, processor, samples,
                                max_tokens=args.max_tokens, verbose=args.verbose,
                                prompt_template=args.prompt,
                                generation_mode=args.generation_mode)

    print("\n=== Aggregates ===")
    for k, v in aggs.items():
        print(f"  {k}: {v}")

    out_dir = OUTPUTS_DIR / f"{BASE_NAME}-{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refcoco_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")

    # Merge into summary.json so cards.py / report.py pick it up automatically.
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"label": args.label, "model_path": args.model, "benchmarks": {}}
    summary.setdefault("benchmarks", {})["refcoco"] = {
        "accuracy": aggs["accuracy_50"],
        "accuracy_75": aggs["accuracy_75"],
        "mean_iou": aggs["mean_iou"],
        "parse_fail_rate": aggs["parse_fail_rate"],
        "n": aggs["n"],
    }
    # Note: per-sample tps values from mlx-vlm's GenerationResult are unreliable
    # for very short outputs (single-token decodes blow up the average). Use
    # measure_perf.py for steady-state perf numbers instead — we deliberately
    # don't write avg_decode_tps to summary["perf"] here.
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote → {summary_path}")
    print(f"Predictions → {out_dir / 'refcoco_predictions.jsonl'}")


if __name__ == "__main__":
    main()
