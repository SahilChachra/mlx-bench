"""
Thin wrapper around mlx-optiq for per-layer KL-sensitivity mixed-precision
quantization, plus the small bit of glue needed to promote optiq's nested
output layout (`<out>/optiq_mixed/`) to a flat variant directory the rest of
the pipeline expects.

Usage:
  python -m scripts.quantization.optiq --target-bpw 5.0 --candidate-bits 3,4,6,8
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_NAME, MODALITY

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"
DATASETS_DIR = REPO / "datasets"
BASE_MODEL = str(MODELS_DIR / f"{BASE_NAME}-fp16")
DEFAULT_CAL = DATASETS_DIR / "optiq_calibration.jsonl"


def promote_optiq_mixed(out_dir: Path):
    """Move <out>/optiq_mixed/* up to <out>/ and prune the other tier dirs."""
    nested = out_dir / "optiq_mixed"
    if not nested.exists():
        print(f"  (no optiq_mixed/ subdir at {nested}; skipping promote)")
        return
    for entry in nested.iterdir():
        target = out_dir / entry.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(entry), str(target))
    shutil.rmtree(nested)
    for sib in ("static_mixed", "uniform_4bit"):
        d = out_dir / sib
        if d.exists():
            shutil.rmtree(d)
    print(f"  promoted optiq_mixed/ → {out_dir}")


def main():
    if MODALITY == "vlm":
        sys.exit("OptiQ is mlx-lm only — no VLM support upstream yet. "
                 "Use affine / mxfp4 / mxfp8 for VLM workflows.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-bpw", type=float, required=True)
    ap.add_argument("--candidate-bits", default="3,4,6,8")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--calibration", default=str(DEFAULT_CAL))
    ap.add_argument("--label", default=None,
                    help="Variant label (default: optiq-<bpw>bpw)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Pass-through args to optiq convert")
    args = ap.parse_args()

    label = args.label or f"optiq-{args.target_bpw:g}bpw".replace(".", "_")
    out = MODELS_DIR / f"{BASE_NAME}-{label}"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "optiq", "convert",
        BASE_MODEL,
        "--output", str(out),
        "--target-bpw", str(args.target_bpw),
        "--candidate-bits", args.candidate_bits,
        "--group-size", str(args.group_size),
        "--calibration-mix", args.calibration,
    ]
    if args.extra:
        cmd.extend(args.extra)
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)
    promote_optiq_mixed(out)
    print(f"OptiQ variant ready → {out}")


if __name__ == "__main__":
    main()
