#!/usr/bin/env bash
#
# Run benchmark.py once per model in a fresh Python process.
# Each model gets a clean process → peak memory is measured accurately.
# 2-minute cooldown between runs to let the GPU settle.
#
# Usage:
#   bash scripts/run_all_isolated.sh
#
set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
VENV="/Users/sahil/venv/mlx"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"
COOLDOWN_SECONDS=120

# Base name and benchmark script can be overridden via env so the same runner
# works for different models / benchmark suites.
#   MLX_BENCH_BASE_NAME=hy-mt2-7b MLX_BENCH_SCRIPT=scripts/flores_benchmark.py bash scripts/run_all_isolated.sh
BENCH_SCRIPT="${MLX_BENCH_SCRIPT:-scripts/benchmark.py}"
BASE_NAME="${MLX_BENCH_BASE_NAME:-granite-4.1-8b}"

# ── activate venv ─────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$PROJECT_DIR"

# ── discover models ───────────────────────────────────────────────────────────
MODELS=()
for d in "$MODELS_DIR"/${BASE_NAME}-*/; do
    [[ -d "$d" ]] && MODELS+=("${d%/}")
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No models found in $MODELS_DIR/"
    exit 1
fi

echo "================================================================"
echo "Isolated benchmark run — ${#MODELS[@]} models, ${COOLDOWN_SECONDS}s cooldown between each"
echo "================================================================"
for m in "${MODELS[@]}"; do
    echo "  - $(basename "$m")"
done
echo

START_TIME=$(date +%s)

# ── run each model in its own process ─────────────────────────────────────────
for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    model_name="$(basename "$model_path")"
    # Strip the "granite-4.1-8b-" prefix (if present) for a clean --label
    label="${model_name#${BASE_NAME}-}"

    echo
    echo "================================================================"
    echo "[$((i+1))/${#MODELS[@]}] $model_name  (label: $label)"
    echo "================================================================"

    python "$BENCH_SCRIPT" --model "$model_path" --label "$label"

    # Cooldown (skip after last model)
    if [[ $i -lt $((${#MODELS[@]} - 1)) ]]; then
        echo
        echo "Cooling down for ${COOLDOWN_SECONDS}s..."
        for remaining in $(seq "$COOLDOWN_SECONDS" -10 10); do
            printf "  %ds remaining...\r" "$remaining"
            sleep 10
        done
        echo "  Done. Starting next model.    "
    fi
done

# ── done ──────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo
echo "================================================================"
echo "All ${#MODELS[@]} models benchmarked in ${HOURS}h ${MINUTES}m"
echo "================================================================"
echo "Results in: $PROJECT_DIR/outputs/"
echo "Next steps:"
echo "  python scripts/generate_model_cards.py"
echo "  python scripts/generate_report.py"
