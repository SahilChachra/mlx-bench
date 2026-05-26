# Pipeline walkthrough

How to take a new HuggingFace model from FP16 download to published HF model
cards. Everything is gated on environment variables so adding a new model
doesn't require code changes — only `export`s and CLI invocations.

## 0. Prerequisites

- Apple Silicon (M1+), MLX installed via `uv pip install mlx-lm`.
- For OptiQ: `uv pip install mlx-optiq`.
- For DWQ: provided by mlx-lm directly (no extra install).
- HuggingFace token in `HF_TOKEN` if you want to push.

## 1. Tell the pipeline which model you're working with

```bash
export MLX_BENCH_BASE_NAME="minicpm5-1b"          # local dir prefix
export MLX_BENCH_HF_REPO="openbmb/MiniCPM5-1B"    # source repo
export MLX_BENCH_DISPLAY_NAME="MiniCPM5-1B"       # used in cards (optional)
```

These are read by `scripts/config.py` and bubble through every script.

## 2. Download the FP16 baseline

```bash
huggingface-cli download "$MLX_BENCH_HF_REPO" --local-dir "models/${MLX_BENCH_BASE_NAME}-fp16"
```

The pipeline expects FP16 at `models/<base>-fp16/`. The rest auto-discovers.

## 3. Build the datasets

```bash
python -m scripts.benchmarks.setup_datasets thinking   # MATH-500 + IFEval
python -m scripts.benchmarks.build_optiq_calibration   # only if running OptiQ
```

Use `classic` for the original GSM8K/MMLU/HumanEval/long-ctx suite, or `all`.

## 4. Quantize

Pick one or more methods:

```bash
# affine: any combination of bits + modes
python -m scripts.pipeline quantize --method affine --bits 4 8 --verify

# DWQ (good at 4-bit, pointless at 8-bit)
python -m scripts.pipeline quantize --method dwq --bits 4

# OptiQ mixed-precision (target a bpw budget)
python -m scripts.pipeline quantize --method optiq --target-bpw 5.0 \
  --candidate-bits 3,4,6,8
```

Variants land in `models/<base>-<label>/`.

## 5. Benchmark

```bash
# everything in models/, in isolated subprocesses with cooldown
bash scripts/run_all_isolated.sh

# or a single variant
python -m scripts.benchmarks.runner --model models/${MLX_BENCH_BASE_NAME}-8bit --label 8bit

# steady-state perf (separate from quality run — important for cards)
python -m scripts.benchmarks.measure_perf --model models/${MLX_BENCH_BASE_NAME}-8bit --label 8bit
```

Each run writes `outputs/<base>-<label>/summary.json`.

## 6. Cards + cross-variant report

```bash
python -m scripts.pipeline card     # writes models/<variant>/README.md
python -m scripts.pipeline report   # writes reports/<base>_benchmark.md
```

Cards auto-pick up `optiq_metadata.json` (for OptiQ variants) and add the
per-layer histogram + sensitivity-preserved layers section.

## 7. Push to HuggingFace

```bash
HF_TOKEN=... python -m scripts.pipeline push                  # all variants
HF_TOKEN=... python -m scripts.pipeline push --only 8bit      # one variant
```

Repo name: `sahilchachra/<base>-<variant>-mlx`.

## All in one shot

```bash
python -m scripts.pipeline all --method affine --push
```

…runs quantize → bench (via `run_all_isolated.sh`) → card → report → push.

## Adding a new quantization method

1. Add `scripts/quantization/<method>.py` with the same shape as
   `dwq.py` / `optiq.py` — argparse + a `subprocess.run` to whatever upstream
   tool does the work, plus any post-step layout normalization.
2. Add the method to `scripts/pipeline.py`'s `quantize` subcommand choices.
3. Add a section to `docs/QUANTIZATION_METHODS.md`.

Card generation does not need updating unless your method emits a metadata
file you want to surface (see how OptiQ's `optiq_metadata.json` is consumed in
`publish/cards.py`).
