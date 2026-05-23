#!/usr/bin/env bash
# Full pipeline: setup → FP16 baseline → quantize → benchmark → report
#
# Generic over any base model — set these env vars before running:
#   MLX_BENCH_BASE_NAME      e.g. granite-4.1-8b   (used in folder naming)
#   MLX_BENCH_HF_REPO        e.g. ibm-granite/granite-4.1-8b
#   MLX_BENCH_DISPLAY_NAME   optional; defaults to BASE_NAME
#
# Optionally override which benchmark script to run (default: scripts/benchmark.py):
#   MLX_BENCH_SCRIPT=scripts/flores_benchmark.py
#
# Usage:
#   ./scripts/run_pipeline.sh                # full run with current env
#   ./scripts/run_pipeline.sh --bits 4 8     # only quantize 4bit and 8bit
#   ./scripts/run_pipeline.sh --skip-fp16    # skip FP16 baseline (already benchmarked)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

BITS="4 5 6 8"
SKIP_FP16=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --bits) BITS="$2"; shift 2 ;;
    --skip-fp16) SKIP_FP16=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${MLX_BENCH_BASE_NAME:-}" || -z "${MLX_BENCH_HF_REPO:-}" ]]; then
    echo "ERROR: set MLX_BENCH_BASE_NAME and MLX_BENCH_HF_REPO before running."
    echo "  e.g. export MLX_BENCH_BASE_NAME=granite-4.1-8b"
    echo "       export MLX_BENCH_HF_REPO=ibm-granite/granite-4.1-8b"
    exit 1
fi

BENCH_SCRIPT="${MLX_BENCH_SCRIPT:-scripts/benchmark.py}"
DISPLAY="${MLX_BENCH_DISPLAY_NAME:-$MLX_BENCH_BASE_NAME}"

echo "=================================================="
echo "  ${DISPLAY} Quantization Pipeline"
echo "=================================================="
echo "  Base name : $MLX_BENCH_BASE_NAME"
echo "  HF repo   : $MLX_BENCH_HF_REPO"
echo "  Bench     : $BENCH_SCRIPT"
echo "  Bits      : $BITS"
echo "  Skip FP16 : $SKIP_FP16"
echo ""

cd "$ROOT"

# 1. Datasets (idempotent — skip if you don't need text-generation evals)
echo "[1/4] Setting up datasets..."
python scripts/setup_datasets.py || echo "  (setup_datasets failed or skipped — OK if your benchmark script doesn't need them)"

# 2. FP16 baseline
if [ "$SKIP_FP16" = false ]; then
  echo ""
  echo "[2/4] FP16 baseline benchmark..."
  python "$BENCH_SCRIPT" --model "models/${MLX_BENCH_BASE_NAME}-fp16" --label fp16
else
  echo "[2/4] Skipping FP16 baseline."
fi

# 3. Quantize
echo ""
echo "[3/4] Quantizing..."
for bits in $BITS; do
  python scripts/quantize.py --bits "$bits" --verify
done

# 4. Benchmark quantized models in isolated processes
echo ""
echo "[4/4] Benchmarking quantized models..."
MLX_BENCH_SCRIPT="$BENCH_SCRIPT" bash scripts/run_all_isolated.sh

# Report
echo ""
echo "[Report] Generating final benchmark report..."
python scripts/generate_report.py

echo ""
echo "=================================================="
echo "  Pipeline complete."
echo "  Report: reports/${MLX_BENCH_BASE_NAME}_benchmark.md"
echo "=================================================="
