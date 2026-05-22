#!/usr/bin/env bash
# Full pipeline: setup → FP16 baseline → quantize → benchmark → evaluate → report
# Run from the MLX-Quantisation/ directory.
#
# Usage:
#   ./scripts/run_pipeline.sh              # full run
#   ./scripts/run_pipeline.sh --bits 4 6   # only quantize 4bit and 6bit
#   ./scripts/run_pipeline.sh --skip-fp16  # skip FP16 baseline (if already done)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

BITS="4 5 6 8"
SKIP_FP16=false

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --bits) BITS="$2"; shift 2 ;;
    --skip-fp16) SKIP_FP16=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=================================================="
echo "  Granite 4.1 8B Quantization Pipeline"
echo "=================================================="
echo "  Bits: $BITS"
echo "  Skip FP16: $SKIP_FP16"
echo ""

cd "$ROOT"

# 1. Datasets
echo "[1/5] Setting up datasets..."
python scripts/setup_datasets.py

# 2. FP16 baseline
if [ "$SKIP_FP16" = false ]; then
  echo ""
  echo "[2/5] FP16 baseline benchmark..."
  python scripts/benchmark.py --model ibm-granite/granite-4.1-8b --label fp16

  echo ""
  echo "[2/5] FP16 baseline evaluation..."
  python scripts/evaluate.py --model ibm-granite/granite-4.1-8b --label fp16
else
  echo "[2/5] Skipping FP16 baseline."
fi

# 3. Quantize
echo ""
echo "[3/5] Quantizing..."
for bits in $BITS; do
  python scripts/quantize.py --bits "$bits" --verify
done

# 4. Benchmark all quantized models
echo ""
echo "[4/5] Benchmarking all quantized models..."
python scripts/benchmark.py --all

# 5. Evaluate all quantized models
echo ""
echo "[5/5] Evaluating all quantized models..."
python scripts/evaluate.py --all

# Report
echo ""
echo "[Report] Generating final benchmark report..."
python scripts/generate_report.py

echo ""
echo "=================================================="
echo "  Pipeline complete."
echo "  Report: reports/final_benchmark.md"
echo "=================================================="
