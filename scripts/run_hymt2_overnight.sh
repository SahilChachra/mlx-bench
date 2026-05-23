#!/usr/bin/env bash
#
# Hy-MT2-7B overnight pipeline:
#   1. Quantize fp16 → 4-bit and 8-bit
#   2. Benchmark all 3 variants (FLORES + perf) in isolated processes
#   3. Generate model cards
#   4. Push 4-bit and 8-bit to HuggingFace
#
# Single log file: reports/hymt2_overnight.log
#
# Usage:
#   bash scripts/run_hymt2_overnight.sh
#
set -uo pipefail   # no -e: we want the script to continue past per-step errors

VENV="/Users/sahil/venv/mlx"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p reports

source "$VENV/bin/activate"

export MLX_BENCH_BASE_NAME="hy-mt2-7b"
export MLX_BENCH_HF_REPO="tencent/Hy-MT2-7B"
export MLX_BENCH_DISPLAY_NAME="Hy-MT2-7B"
export MLX_BENCH_SCRIPT="scripts/flores_benchmark.py"

LOG="reports/hymt2_overnight.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "Hy-MT2-7B overnight pipeline — $(date)"
echo "================================================================"

# ── Step 1: Quantize ──────────────────────────────────────────────────────────
echo
echo "── Step 1/4: Quantize ──"
python scripts/quantize.py --bits 4 8
echo "Quantization done — $(date)"

# ── Step 2: Benchmarks ────────────────────────────────────────────────────────
echo
echo "── Step 2/4: Benchmarks (isolated processes) ──"
COOLDOWN_SECONDS=60
VARIANTS=(fp16 8bit 4bit)
for i in "${!VARIANTS[@]}"; do
    label="${VARIANTS[$i]}"
    model_path="models/hy-mt2-7b-${label}"
    if [[ ! -d "$model_path" ]]; then
        echo "  SKIP $label — folder $model_path not found"
        continue
    fi
    echo
    echo "[$((i+1))/${#VARIANTS[@]}] Benchmarking $label"
    python scripts/flores_benchmark.py --model "$model_path" --label "$label" --n-per-pair 20
    if [[ $i -lt $((${#VARIANTS[@]} - 1)) ]]; then
        echo "Cooling down ${COOLDOWN_SECONDS}s..."
        sleep "$COOLDOWN_SECONDS"
    fi
done
echo "Benchmarks done — $(date)"

# ── Step 3: Model cards ───────────────────────────────────────────────────────
echo
echo "── Step 3/4: Generate model cards ──"
python scripts/generate_model_cards_hymt2.py
echo "Cards done — $(date)"

# ── Step 4: Push to HuggingFace ───────────────────────────────────────────────
echo
echo "── Step 4/4: Push to HuggingFace ──"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set — skipping push"
else
    # Strip extra dirs that shouldn't ship to HF
    for v in 4bit 8bit; do
        rm -rf "models/hy-mt2-7b-${v}/train" "models/hy-mt2-7b-${v}/imgs" 2>/dev/null || true
    done
    python scripts/push_to_hf.py --only 4bit 8bit
fi
echo "Push done — $(date)"

echo
echo "================================================================"
echo "Pipeline complete — $(date)"
echo "================================================================"
