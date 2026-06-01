"""
Quantize the base model into target bit-widths, mixed-bit recipes,
block float modes, and group size variants.

Model naming convention:
  granite-4.1-8b-4bit          → affine 4-bit, group 64 (default)
  granite-4.1-8b-4bit-g32      → affine 4-bit, group 32
  granite-4.1-8b-4bit-g128     → affine 4-bit, group 128
  granite-4.1-8b-mixed4_6      → mixed 4+6 bit
  granite-4.1-8b-mxfp4         → block float MX FP4
  granite-4.1-8b-mxfp8         → block float MX FP8

Usage:
  python scripts/quantize.py --bits 4 6 8
  python scripts/quantize.py --bits 4 --group-sizes 32 128
  python scripts/quantize.py --mixed 4_6
  python scripts/quantize.py --q-mode mxfp4 mxfp8
  python scripts/quantize.py --bits 4 --group-sizes 32 128 --q-mode mxfp4 mxfp8 --mixed 4_6 --verify
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_NAME, BASE_HF_REPO, MODALITY

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)
# LLM workflow expects a local FP16 dir; VLM workflow points mlx-vlm directly
# at the HF repo (mlx-vlm fetches and converts in one pass).
BASE_MODEL = BASE_HF_REPO if MODALITY == "vlm" else str(MODELS_DIR / f"{BASE_NAME}-fp16")

ALL_BITS        = [4, 5, 6, 8]
DEFAULT_GROUP   = 64

# Backend selection — both expose `convert` with --hf-path / --mlx-path /
# -q / --q-bits / --q-group-size / --q-mode / --quant-predicate.
_CONVERT_MOD = "mlx_vlm" if MODALITY == "vlm" else "mlx_lm"
# Many VLMs (e.g. LocateAnything) declare auto_map → custom modeling code.
_EXTRA_CONVERT_FLAGS = ["--trust-remote-code"] if MODALITY == "vlm" else []


def _run(cmd, output_path):
    print(f"  Command: {' '.join(cmd)}\n")
    t_start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t_start

    if result.returncode == 0:
        size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1024**2
        print(f"\n  Done in {elapsed:.0f}s | Disk: {size_mb:.0f} MB")
        return True
    else:
        print(f"\n  ERROR: exit code {result.returncode}")
        return False


def _skip_or_run(output_path, cmd):
    if output_path.exists() and any(output_path.iterdir()):
        print(f"  Already exists, skipping → {output_path.name}")
        print(f"  (delete to re-run: rm -rf {output_path})")
        return True
    return _run(cmd, output_path)


def quantize_uniform(bits, group=DEFAULT_GROUP, upload_prefix=None):
    g_suffix = f"-g{group}" if group != DEFAULT_GROUP else ""
    name = f"{BASE_NAME}-{bits}bit{g_suffix}"
    output_path = MODELS_DIR / name

    print(f"\n{'='*60}")
    print(f"Affine {bits}-bit  group={group} → {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", _CONVERT_MOD, "convert",
        *_EXTRA_CONVERT_FLAGS,
        "--hf-path", BASE_MODEL,
        "--mlx-path", str(output_path),
        "-q", "--q-bits", str(bits),
        "--q-group-size", str(group),
    ]
    if upload_prefix:
        repo = f"{upload_prefix}/{name}-mlx"
        cmd += ["--upload-repo", repo]
        print(f"  Upload → {repo}")

    return _skip_or_run(output_path, cmd)


def quantize_mixed(recipe, upload_prefix=None):
    name = f"{BASE_NAME}-mixed{recipe}"
    output_path = MODELS_DIR / name

    print(f"\n{'='*60}")
    print(f"Mixed-bit {recipe} → {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", _CONVERT_MOD, "convert",
        *_EXTRA_CONVERT_FLAGS,
        "--hf-path", BASE_MODEL,
        "--mlx-path", str(output_path),
        "-q", "--quant-predicate", f"mixed_{recipe}",
    ]
    if upload_prefix:
        repo = f"{upload_prefix}/{name}-mlx"
        cmd += ["--upload-repo", repo]
        print(f"  Upload → {repo}")

    return _skip_or_run(output_path, cmd)


def quantize_mode(mode, upload_prefix=None):
    """Block float modes: mxfp4, mxfp8, nvfp4."""
    name = f"{BASE_NAME}-{mode}"
    output_path = MODELS_DIR / name

    print(f"\n{'='*60}")
    print(f"Block-float {mode} → {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", _CONVERT_MOD, "convert",
        *_EXTRA_CONVERT_FLAGS,
        "--hf-path", BASE_MODEL,
        "--mlx-path", str(output_path),
        "-q", "--q-mode", mode,
    ]
    if upload_prefix:
        repo = f"{upload_prefix}/{name}-mlx"
        cmd += ["--upload-repo", repo]
        print(f"  Upload → {repo}")

    return _skip_or_run(output_path, cmd)


def verify(model_path):
    print(f"\nVerifying {model_path.name}...")
    try:
        if MODALITY == "vlm":
            from mlx_vlm import load, generate
            model, processor = load(str(model_path))
            response = generate(model, processor, prompt="Hello, briefly introduce yourself.",
                                image=None, max_tokens=50, verbose=False)
        else:
            from mlx_lm import load, generate
            model, tokenizer = load(str(model_path))
            response = generate(model, tokenizer, prompt="Hello, briefly introduce yourself.",
                                max_tokens=50, verbose=False)
        print(f"  OK — {str(response)[:100]}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", nargs="+", type=int, choices=ALL_BITS,
                        help="Uniform affine bit-widths (4 5 6 8)")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[DEFAULT_GROUP],
                        metavar="G", help=f"Quantization group sizes (default: {DEFAULT_GROUP})")
    parser.add_argument("--mixed", nargs="+", choices=["4_6", "3_4", "3_6", "2_6"],
                        help="Mixed-bit recipes")
    parser.add_argument("--q-mode", nargs="+", choices=["mxfp4", "mxfp8", "nvfp4"],
                        help="Block float modes")
    parser.add_argument("--all", action="store_true",
                        help="Run all uniform bits (4/5/6/8) at default group size")
    parser.add_argument("--verify", action="store_true",
                        help="Smoke test each model after quantization")
    parser.add_argument("--upload-prefix", type=str,
                        help="HF username for uploads")
    args = parser.parse_args()

    bit_targets   = ALL_BITS if args.all else (args.bits or [])
    group_targets = args.group_sizes or [DEFAULT_GROUP]
    mixed_targets = args.mixed or []
    mode_targets  = args.q_mode or []

    if not bit_targets and not mixed_targets and not mode_targets:
        parser.print_help()
        sys.exit(1)

    print(f"Uniform bits  : {bit_targets or '—'}")
    print(f"Group sizes   : {group_targets}")
    print(f"Mixed recipes : {mixed_targets or '—'}")
    print(f"Block-float   : {mode_targets or '—'}")
    print(f"Base model    : {BASE_MODEL}")

    results = {}

    for bits in bit_targets:
        for group in group_targets:
            g_suffix = f"-g{group}" if group != DEFAULT_GROUP else ""
            key = f"{bits}bit{g_suffix}"
            results[key] = quantize_uniform(bits, group, args.upload_prefix)
            if results[key] and args.verify:
                g_sfx = f"-g{group}" if group != DEFAULT_GROUP else ""
                results[f"{key}_verify"] = verify(MODELS_DIR / f"{BASE_NAME}-{bits}bit{g_sfx}")

    for recipe in mixed_targets:
        key = f"mixed{recipe}"
        results[key] = quantize_mixed(recipe, args.upload_prefix)
        if results[key] and args.verify:
            results[f"{key}_verify"] = verify(MODELS_DIR / f"{BASE_NAME}-mixed{recipe}")

    for mode in mode_targets:
        results[mode] = quantize_mode(mode, args.upload_prefix)
        if results[mode] and args.verify:
            results[f"{mode}_verify"] = verify(MODELS_DIR / f"{BASE_NAME}-{mode}")

    print(f"\n{'='*60}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*60}")
    for key, ok in results.items():
        if "_verify" not in key:
            v = results.get(f"{key}_verify")
            v_str = f" | verify: {'OK' if v else 'FAILED'}" if args.verify else ""
            print(f"  {key}: {'OK' if ok else 'FAILED'}{v_str}")
