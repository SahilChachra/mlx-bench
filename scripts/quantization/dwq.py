"""
Thin wrapper around mlx_lm's distillation-aware weight quantization (DWQ).

DWQ tunes per-group scales/zeros against the FP16 teacher via KL divergence.
Practical observation from our MiniCPM5 runs: DWQ only helps meaningfully at
low bits (≤4); at 8-bit the naive quant already matches the FP16 KL noise
floor, so DWQ silently aborts.

Usage:
  python -m scripts.quantization.dwq --bits 4 --group-size 64
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_NAME

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"
BASE_MODEL = str(MODELS_DIR / f"{BASE_NAME}-fp16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=1024)
    ap.add_argument("--out", default=None,
                    help=f"Output dir (default: models/{BASE_NAME}-dwq-Nbit)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Pass-through args to mlx_lm.quant.dwq")
    args = ap.parse_args()

    out = args.out or str(MODELS_DIR / f"{BASE_NAME}-dwq-{args.bits}bit")
    cmd = [
        sys.executable, "-m", "mlx_lm.quant.dwq",
        "--model", BASE_MODEL,
        "--mlx-path", out,
        "--bits", str(args.bits),
        "--group-size", str(args.group_size),
        "--num-samples", str(args.num_samples),
    ]
    if args.extra:
        cmd.extend(args.extra)
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote DWQ {args.bits}-bit → {out}")


if __name__ == "__main__":
    main()
