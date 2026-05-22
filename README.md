# mlx-bench

A reproducible benchmarking pipeline for **MLX-quantized LLMs** on Apple Silicon.

Quantize any HuggingFace LLM that [mlx-lm](https://github.com/ml-explore/mlx-lm) supports into every available variant (4/5/6/8-bit affine, mixed-bit, block-float MX FP4/FP8), then measure performance **and** quality side-by-side in a single pass — so you can pick the right variant for your use case.

**Runtime**: MLX + mlx-lm  
**Hardware**: Any Apple Silicon Mac (M1 or later)

---

## What gets measured

For each model variant, in a single load:

### Quality (uses public benchmark datasets)

| Benchmark | Task | Metric | Samples |
|---|---|---|---|
| GSM8K | Math word problems | Accuracy (exact answer match) | 25 |
| HumanEval | Python code generation | pass@1 (tests pass) | 20 |
| MMLU | Multiple choice world knowledge | Accuracy | 50 |
| Long-context | Long-form generation | Length requirement met | 10 |

### Performance (from mlx-lm's `GenerationResponse`)

- **Prefill tok/s** — prompt processing speed
- **Decode tok/s** — generation speed
- **Peak memory (GB)** — Metal/unified memory peak
- **Context scaling** — decode speed at 128 / 256 / 512 / 1024 token contexts

> Sample sizes are intentionally small (~100 total) for fast iteration. Treat absolute accuracy numbers as **indicative**, not definitive. Cross-variant deltas are reliable.

---

## Quantization variants

| Variant | Method | Notes |
|---|---|---|
| `4bit` / `5bit` / `6bit` / `8bit` | Affine integer | Standard integer quantization. Group size 64 by default. |
| `mixed4_6` (and other mixed recipes) | Mixed-bit | Sensitive layers (embeddings, first/last layers) at higher precision, rest at lower. Better quality than uniform 4-bit at similar size. |
| `mxfp4` / `mxfp8` | Block float (Microscaling) | Floating-point representation per block instead of integer. Different quality/speed profile vs affine at same bit-width. |
| Custom group sizes | Affine | `--group-sizes 32 128` etc. — affects quality vs compression tradeoff. |

All variants are produced by `mlx_lm convert` under the hood — no custom quantization code.

---

## Setup

```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install mlx-lm datasets huggingface_hub tqdm
```

---

## Usage

### 1. Download benchmark datasets (one-time)

```bash
python scripts/setup_datasets.py
```

### 2. Download a base model (FP16 / BF16)

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="<HF_REPO_ID>",                       # e.g. ibm-granite/granite-4.1-8b
    local_dir="./models/<MODEL_NAME>-fp16",       # e.g. ./models/granite-4.1-8b-fp16
    max_workers=8,
)
```

### 3. Quantize

Pointing `--bits / --mixed / --q-mode` at `scripts/quantize.py` runs `mlx_lm convert` and writes variants into `./models/`.

```bash
# All uniform variants (4/5/6/8 bit affine)
python scripts/quantize.py --all --verify

# Mixed-bit (e.g. 4-bit body, 6-bit sensitive layers)
python scripts/quantize.py --mixed 4_6 --verify

# Block-float (microscaling)
python scripts/quantize.py --q-mode mxfp4 mxfp8 --verify

# Custom group sizes (e.g. 4-bit at group 32 and 128)
python scripts/quantize.py --bits 4 --group-sizes 32 128 --verify
```

> Update `BASE_MODEL` in `scripts/quantize.py` to point at your FP16 model folder, and adjust the output naming if you want something other than the default `<model>-<variant>` convention.

### 4. Benchmark a model

```bash
python scripts/benchmark.py --model ./models/<MODEL_NAME>-<VARIANT> --label <VARIANT>
```

Or benchmark every quantized variant in `./models/` sequentially with a 2-minute cooldown between runs:

```bash
python scripts/benchmark.py --all
```

### 5. Generate the comparison report

```bash
python scripts/generate_report.py        # → reports/final_benchmark.md
```

### 6. (Optional) Generate HuggingFace model cards

For publishing quantized variants to HuggingFace, generate per-model README cards from the benchmark results:

```bash
python scripts/generate_model_cards.py    # writes models/<variant>/README.md
```

> Update the model-name / author constants at the top of `generate_model_cards.py` to match your setup.

---

## End-to-end pipeline

```bash
source .venv/bin/activate

python scripts/setup_datasets.py
python scripts/quantize.py --all --mixed 4_6 --q-mode mxfp4 mxfp8 --verify
python scripts/benchmark.py --all
python scripts/generate_report.py
```

---

## Project structure

```
.
├── models/                          # Downloaded + quantized models (git-ignored)
│   ├── <model-name>-fp16/
│   ├── <model-name>-4bit/
│   ├── <model-name>-mxfp4/
│   └── ...
│
├── datasets/                        # Benchmark prompts (JSONL)
│   ├── gsm8k.jsonl
│   ├── humaneval.jsonl
│   ├── mmlu.jsonl
│   └── long_context_prompts.jsonl
│
├── outputs/                         # Per-model results (git-ignored)
│   └── <model-name>-<variant>/
│       ├── gsm8k.jsonl              # Per-sample results
│       ├── humaneval.jsonl
│       ├── mmlu.jsonl
│       ├── long_context.jsonl
│       ├── context_scaling.json
│       └── summary.json
│
├── reports/                         # Generated reports (git-ignored)
│   └── final_benchmark.md
│
└── scripts/
    ├── setup_datasets.py            # Download benchmark datasets
    ├── quantize.py                  # Quantize via mlx_lm convert
    ├── benchmark.py                 # Run benchmarks (perf + quality)
    ├── generate_report.py           # Compile comparison report
    └── generate_model_cards.py      # Generate HF README cards
```

---

## Notes

- All inference uses mlx-lm's `stream_generate`. No custom inference code.
- All quantization uses `mlx_lm convert` under the hood. No custom quant code.
- Benchmarks measure both per-sample performance (prefill/decode tok/s, peak memory) **and** quality (accuracy / pass@1) in a single pass — the model is loaded once per variant.

---

## License

Apache 2.0
