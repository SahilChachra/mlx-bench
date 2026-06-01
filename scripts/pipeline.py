"""
Unified entry point for the mlx-bench pipeline.

Subcommands:
  quantize  --method {affine,dwq,optiq}   ...method-specific flags...
  bench     --label <variant>             [--script runner|flores|measure_perf]
  card      [--only <variant>]
  report    [--out reports/x.md]
  push      [--only <variant>] [--skip ...]
  all       --method ...                  end-to-end: quantize → bench → card → push

Examples:
  python -m scripts.pipeline quantize --method affine --bits 4 8
  python -m scripts.pipeline quantize --method optiq --target-bpw 5.0
  python -m scripts.pipeline bench --label optiq-5_0bpw
  python -m scripts.pipeline card
  python -m scripts.pipeline push --only optiq-5_0bpw

All scripts read MLX_BENCH_BASE_NAME / MLX_BENCH_HF_REPO / MLX_BENCH_DISPLAY_NAME
from the environment (see scripts/config.py).
"""

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable
REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_NAME, BASE_HF_REPO, MODALITY


def run(cmd):
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


# ── quantize ──────────────────────────────────────────────────────────────────
def cmd_quantize(args, rest):
    method = args.method
    if method == "affine":
        run([PY, "-m", "scripts.quantization.affine", *rest])
    elif method == "dwq":
        run([PY, "-m", "scripts.quantization.dwq", *rest])
    elif method == "optiq":
        run([PY, "-m", "scripts.quantization.optiq", *rest])
    else:
        raise SystemExit(f"unknown method: {method}")


# ── bench ─────────────────────────────────────────────────────────────────────
BENCH_SCRIPTS = {
    "runner":       "scripts.benchmarks.runner",
    "flores":       "scripts.benchmarks.flores",
    "measure_perf": "scripts.benchmarks.measure_perf",
}


def cmd_bench(args, rest):
    mod = BENCH_SCRIPTS.get(args.script, args.script)
    run([PY, "-m", mod, *rest])


# ── publish ───────────────────────────────────────────────────────────────────
def cmd_card(args, rest):
    run([PY, "-m", "scripts.publish.cards", *rest])


def cmd_report(args, rest):
    run([PY, "-m", "scripts.publish.report", *rest])


def cmd_push(args, rest):
    run([PY, "-m", "scripts.publish.push", *rest])


# ── all ───────────────────────────────────────────────────────────────────────
def cmd_all(args, rest):
    cmd_quantize(args, rest)
    # bench every variant via run_all_isolated.sh (handles per-variant subprocess isolation)
    run(["bash", str(REPO / "scripts" / "run_all_isolated.sh")])
    cmd_card(args, [])
    cmd_report(args, [])
    if args.push:
        cmd_push(args, [])


def main():
    print(f"[pipeline] modality={MODALITY}  base={BASE_NAME}  repo={BASE_HF_REPO}")
    ap = argparse.ArgumentParser(prog="pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="quantize the FP16 base model")
    q.add_argument("--method", choices=["affine", "dwq", "optiq"], required=True)
    q.set_defaults(fn=cmd_quantize)

    b = sub.add_parser("bench", help="run benchmark on a variant")
    b.add_argument("--script", default="runner",
                   help="runner | flores | measure_perf | <dotted.module.path>")
    b.set_defaults(fn=cmd_bench)

    c = sub.add_parser("card", help="generate HF model cards")
    c.set_defaults(fn=cmd_card)

    r = sub.add_parser("report", help="generate cross-variant benchmark report")
    r.set_defaults(fn=cmd_report)

    p = sub.add_parser("push", help="push variant repos to HuggingFace")
    p.set_defaults(fn=cmd_push)

    a = sub.add_parser("all", help="quantize → bench → card → report [→ push]")
    a.add_argument("--method", choices=["affine", "dwq", "optiq"], required=True)
    a.add_argument("--push", action="store_true")
    a.set_defaults(fn=cmd_all)

    args, rest = ap.parse_known_args()
    args.fn(args, rest)


if __name__ == "__main__":
    main()
