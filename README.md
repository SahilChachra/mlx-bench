# mlx-bench

A reproducible benchmarking pipeline for **MLX-quantized LLMs** on Apple Silicon.

Quantize any HuggingFace LLM that [mlx-lm](https://github.com/ml-explore/mlx-lm) supports into every available variant (4/5/6/8-bit affine, mixed-bit, block-float MX FP4/FP8), then measure performance **and** quality side-by-side in a single pass — so you can pick the right variant for your use case.

**Runtime**: MLX + mlx-lm
**Hardware**: Any Apple Silicon Mac (M1 or later)

---

## Design

The pipeline is **model-agnostic**. The model under test is selected via environment variables read by `scripts/config.py`. You should not need to edit the core scripts when adding a new model — just export the right env vars and run.

```bash
export MLX_BENCH_BASE_NAME="granite-4.1-8b"           # local folder prefix
export MLX_BENCH_HF_REPO="ibm-granite/granite-4.1-8b" # source HF repo
export MLX_BENCH_DISPLAY_NAME="Granite 4.1 8B"        # human label (optional)
```

That base name flows through everything: model folders (`models/${BASE_NAME}-{4bit,8bit,...}`), result folders (`outputs/${BASE_NAME}-...`), HF repo names (`<author>/${BASE_NAME}-<variant>-mlx`), and report titles.

### Picking the benchmark suite

Select which benchmark script the runner uses via `MLX_BENCH_SCRIPT`:

```bash
# Text generation (GSM8K / HumanEval / MMLU — default)
export MLX_BENCH_SCRIPT="scripts/benchmark.py"

# Translation (FLORES-200 chrF++ / BLEU)
export MLX_BENCH_SCRIPT="scripts/flores_benchmark.py"
```

`generate_model_cards.py` and `generate_report.py` auto-detect which benchmarks each `summary.json` contains and only render the columns that have data — a model evaluated only on FLORES won't show empty GSM8K columns, and vice versa. When per-pair FLORES data is present, the card includes a per-direction chrF++ table automatically.

If your model needs a fundamentally different evaluation, add a new `scripts/<name>_benchmark.py` that writes a `summary.json` with the same shape (`benchmarks`, `perf`, `context_scaling`) — the cards/report will pick it up.

---

## What gets measured

For each model variant, in a single load:

### Quality (defaults; swap by changing the benchmark script)

| Benchmark | Task | Metric | Provided by |
|---|---|---|---|
| GSM8K     | Math word problems            | Accuracy (exact answer match) | `benchmark.py` |
| HumanEval | Python code generation        | pass@1 (tests pass) | `benchmark.py` |
| MMLU      | Multiple-choice world knowledge | Accuracy | `benchmark.py` |
| Long-context | Long-form generation       | Length requirement met | `benchmark.py` |
| FLORES-200 | Multilingual translation     | chrF++ / BLEU (sacrebleu) | `flores_benchmark.py` |

### Performance (from mlx-lm's `GenerationResponse`)

- **Prefill tok/s** — prompt processing speed
- **Decode tok/s** — generation speed
- **Peak memory (GB)** — Metal/unified memory peak
- **Context scaling** — decode speed at 128 / 256 / 512 / 1024 token contexts

> Sample sizes are intentionally small for fast iteration. Treat absolute accuracy numbers as **indicative**, not definitive. Cross-variant deltas are reliable.

---

## Quantization variants

| Variant | Method | Notes |
|---|---|---|
| `4bit` / `5bit` / `6bit` / `8bit` | Affine integer | Standard integer quantization. Group size 64 by default. |
| `mixed4_6` (and other mixed recipes) | Mixed-bit | Sensitive layers (embeddings, first/last layers) at higher precision, rest at lower. |
| `mxfp4` / `mxfp8` | Block float (Microscaling) | Floating-point representation per block instead of integer. |
| Custom group sizes | Affine | `--group-sizes 32 128` etc. |

All variants are produced by `mlx_lm convert` under the hood — no custom quantization code.

---

## Setup

```bash
uv venv .venv
source .venv/bin/activate

uv pip install mlx-lm datasets huggingface_hub tqdm sacrebleu
```

---

## Usage

### 1. Pick the model

```bash
export MLX_BENCH_BASE_NAME="granite-4.1-8b"
export MLX_BENCH_HF_REPO="ibm-granite/granite-4.1-8b"
```

### 2. Download the FP16 weights

```bash
hf download "$MLX_BENCH_HF_REPO" --local-dir "models/${MLX_BENCH_BASE_NAME}-fp16"
```

### 3. Datasets (one-time, only for text-gen evals)

```bash
python scripts/setup_datasets.py
```

For FLORES-based translation eval, fetch the dataset:

```bash
cd datasets && curl -sLO https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz && tar -xzf flores200_dataset.tar.gz && cd ..
```

### 4. Quantize

```bash
python scripts/quantize.py --bits 4 8                # 4-bit and 8-bit affine
python scripts/quantize.py --all --verify            # 4/5/6/8 + smoke test
python scripts/quantize.py --mixed 4_6 --verify
python scripts/quantize.py --q-mode mxfp4 mxfp8 --verify
```

Folders land in `models/${MLX_BENCH_BASE_NAME}-<variant>/`.

### 5. Benchmark

Single model:

```bash
python scripts/benchmark.py --model "models/${MLX_BENCH_BASE_NAME}-4bit" --label 4bit
```

All variants in isolated processes (clean peak memory per run):

```bash
bash scripts/run_all_isolated.sh
```

To run a non-default benchmark (e.g. translation):

```bash
MLX_BENCH_SCRIPT=scripts/flores_benchmark.py bash scripts/run_all_isolated.sh
```

### 6. Report

```bash
python scripts/generate_report.py   # → reports/${MLX_BENCH_BASE_NAME}_benchmark.md
```

### 7. (Optional) Generate HuggingFace model cards + push

```bash
python scripts/generate_model_cards.py        # writes models/<variant>/README.md
HF_TOKEN=... python scripts/push_to_hf.py     # publishes to <author>/<base>-<variant>-mlx
```

Override the HF author via `MLX_BENCH_HF_AUTHOR` (default: `sahilchachra`).

---

## End-to-end pipeline

`run_pipeline.sh` does: dataset setup → FP16 baseline benchmark → quantize → benchmark each variant (isolated processes) → generate model cards → write report → optionally push to HF.

```bash
source .venv/bin/activate

export MLX_BENCH_BASE_NAME="granite-4.1-8b"
export MLX_BENCH_HF_REPO="ibm-granite/granite-4.1-8b"

bash scripts/run_pipeline.sh
```

For a translation model (FLORES) with mxfp4/mxfp8 variants and HF publish:

```bash
export MLX_BENCH_BASE_NAME="hy-mt2-7b"
export MLX_BENCH_HF_REPO="tencent/Hy-MT2-7B"
export MLX_BENCH_SCRIPT="scripts/flores_benchmark.py"
export HF_TOKEN="..."

bash scripts/run_pipeline.sh --bits "4 8" --q-modes "mxfp4 mxfp8" --skip-fp16 --push
```

Flags: `--bits "<list>"`, `--q-modes "<list>"`, `--mixed "<recipe>"`, `--skip-fp16`, `--push`.

---

## Project structure

```
.
├── models/                          # Downloaded + quantized models (git-ignored)
│   ├── <base-name>-fp16/
│   ├── <base-name>-4bit/
│   └── ...
│
├── datasets/                        # Benchmark prompts (most git-ignored)
│   ├── gsm8k.jsonl
│   ├── humaneval.jsonl
│   ├── mmlu.jsonl
│   ├── long_context_prompts.jsonl
│   └── flores200_dataset/           # downloaded on demand
│
├── outputs/                         # Per-model results (git-ignored)
│   └── <base-name>-<variant>/
│       ├── *.jsonl                  # per-sample records
│       ├── context_scaling.json
│       └── summary.json
│
├── reports/                         # Generated reports (git-ignored)
│
└── scripts/
    ├── config.py                    # Reads MLX_BENCH_* env vars
    ├── setup_datasets.py            # Download text-gen eval datasets
    ├── quantize.py                  # Quantize via mlx_lm convert
    ├── benchmark.py                 # Generic benchmark (GSM8K/HumanEval/MMLU/long-ctx)
    ├── flores_benchmark.py          # Translation benchmark (FLORES-200)
    ├── generate_report.py           # Auto-discovering comparison report
    ├── generate_model_cards.py      # Auto-discovering generic HF cards
    ├── push_to_hf.py                # Publish quantized models to HF
    ├── run_pipeline.sh              # Full pipeline (quantize → bench → cards → report → push)
    └── run_all_isolated.sh          # One isolated process per model variant
```

---

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `MLX_BENCH_BASE_NAME`    | `granite-4.1-8b` | All scripts (folder + repo naming) |
| `MLX_BENCH_HF_REPO`      | `ibm-granite/granite-4.1-8b` | quantize, report, cards |
| `MLX_BENCH_DISPLAY_NAME` | `$BASE_NAME` | report title |
| `MLX_BENCH_SCRIPT`       | `scripts/benchmark.py` | `run_all_isolated.sh`, `run_pipeline.sh` |
| `MLX_BENCH_HF_AUTHOR`    | `sahilchachra` | `generate_model_cards.py`, `push_to_hf.py` |
| `MLX_BENCH_LICENSE`      | `apache-2.0` | `generate_model_cards.py` (frontmatter) |
| `HF_TOKEN`               | — | `push_to_hf.py` |

---

## Notes

- All inference uses mlx-lm's `stream_generate`. No custom inference code.
- All quantization uses `mlx_lm convert` under the hood. No custom quant code.
- Benchmarks measure both per-sample performance (prefill/decode tok/s, peak memory) **and** quality in a single pass — the model is loaded once per variant.
- Peak memory is measured via `mx.get_peak_memory()`, which is process-wide. The runner spawns one process per variant so peaks don't carry over between models.

---

## License

Apache 2.0
