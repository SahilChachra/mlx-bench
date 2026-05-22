# Granite 4.1 8B — MLX Quantization Benchmark

Rigorous quantization benchmarks for IBM's [Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b) on Apple Silicon using the [MLX](https://github.com/ml-explore/mlx) framework.

**Goal**: Understand the quality vs. speed vs. memory tradeoff across every quantization variant available in mlx-lm — so you can pick the right variant for your use case.

**Hardware**: Apple M5 Pro  
**Runtime**: MLX + mlx-lm  
**Base model**: `ibm-granite/granite-4.1-8b` (Apache 2.0)

---

## Quantization Variants

| Model | Method | Bits/weight | Disk |
|---|---|---|---|
| `granite-4.1-8b-fp16` | Baseline (BF16) | 16 | ~17.6 GB |
| `granite-4.1-8b-8bit` | Affine int8 | ~8.5 | ~8.5 GB |
| `granite-4.1-8b-6bit` | Affine int6 | ~6.5 | ~6.5 GB |
| `granite-4.1-8b-5bit` | Affine int5 | ~5.5 | ~5.5 GB |
| `granite-4.1-8b-4bit` | Affine int4 | ~4.5 | ~4.5 GB |
| `granite-4.1-8b-mixed4_6` | Mixed 4+6 bit (sensitive layers at 6-bit) | ~4.77 | ~4.8 GB |
| `granite-4.1-8b-mxfp4` | Block float MX FP4 | ~4 | ~4 GB |
| `granite-4.1-8b-mxfp8` | Block float MX FP8 | ~8 | ~8 GB |

**Affine**: Standard integer quantization (what most tools use by default).  
**Mixed-bit**: Sensitive layers (embeddings, first/last layers) stay at 6-bit; rest at 4-bit. Better quality than uniform 4-bit at similar size.  
**Block float (MX)**: Uses floating-point number representation per block instead of integer. Different quality/speed profile vs affine at the same bit-width.

---

## Benchmarks

Each model is evaluated on **performance** and **quality** in a single pass — no duplicate model loads.

### Quality benchmarks

| Benchmark | Task | Metric | Samples |
|---|---|---|---|
| GSM8K | Math word problems | Accuracy (exact answer match) | 25 |
| HumanEval | Python code generation | pass@1 (tests pass) | 20 |
| MMLU | Multiple choice world knowledge | Accuracy | 50 |
| Long-context | Long-form generation | Length requirement met | 10 |

### Performance metrics (from mlx-lm's `GenerationResponse`)

- **Prefill tok/s** — prompt processing speed
- **Decode tok/s** — generation speed
- **Peak memory (GB)** — Metal/unified memory peak
- **Context scaling** — decode speed at 128 / 256 / 512 / 1024 token contexts

---

## Project Structure

```
MLX-Quantisation/
├── models/                        # Downloaded and quantized models
│   ├── granite-4.1-8b-fp16/       # FP16 baseline (17.6 GB)
│   ├── granite-4.1-8b-4bit/       # Affine 4-bit
│   ├── granite-4.1-8b-5bit/
│   ├── granite-4.1-8b-6bit/
│   ├── granite-4.1-8b-8bit/
│   ├── granite-4.1-8b-mixed4_6/   # Mixed 4+6 bit
│   ├── granite-4.1-8b-mxfp4/      # Block float FP4
│   └── granite-4.1-8b-mxfp8/      # Block float FP8
│
├── datasets/                      # Benchmark datasets (JSONL)
│   ├── gsm8k.jsonl                 # 25 math problems
│   ├── humaneval.jsonl             # 20 code problems
│   ├── mmlu.jsonl                  # 50 multiple choice
│   └── long_context_prompts.jsonl  # 10 long-form prompts
│
├── outputs/                       # Results per model
│   └── granite-4.1-8b-<variant>/
│       ├── gsm8k.jsonl             # Per-sample results
│       ├── humaneval.jsonl
│       ├── mmlu.jsonl
│       ├── long_context.jsonl
│       ├── context_scaling.json    # Scaling curve
│       └── summary.json           # Accuracy + perf summary
│
├── reports/
│   └── final_benchmark.md         # Generated comparison report
│
└── scripts/
    ├── setup_datasets.py           # Download benchmark datasets
    ├── quantize.py                 # Quantize models
    ├── benchmark.py                # Run benchmarks (perf + quality)
    └── generate_report.py          # Compile results into report
```

---

## Setup

```bash
# Create and activate virtual environment
uv venv /Users/sahil/venv/mlx
source /Users/sahil/venv/mlx/bin/activate

# Install dependencies
uv pip install mlx-lm datasets tqdm psutil
```

> **Note**: mlx-lm main branch has a bug fix for Granite tied embeddings (`lm_head.weight`). If you hit a `ValueError: Received 1 parameters not in model: lm_head.weight` error, ensure you're on a version that includes the `sanitize()` fix in `mlx_lm/models/granite.py`.

---

## Commands

### 1. Download datasets

```bash
python scripts/setup_datasets.py
```

### 2. Download base model

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ibm-granite/granite-4.1-8b",
    local_dir="./models/granite-4.1-8b-fp16",
    max_workers=8
)
```

### 3. Quantize

```bash
# All uniform variants
python scripts/quantize.py --all --verify

# Mixed-bit
python scripts/quantize.py --mixed 4_6 --verify

# Block float
python scripts/quantize.py --q-mode mxfp4 mxfp8 --verify

# Group size variants (e.g. 4-bit at group 32 and 128)
python scripts/quantize.py --bits 4 --group-sizes 32 128 --verify
```

### 4. Benchmark a single model

```bash
python scripts/benchmark.py --model ./models/granite-4.1-8b-fp16 --label fp16
python scripts/benchmark.py --model ./models/granite-4.1-8b-4bit --label 4bit
python scripts/benchmark.py --model ./models/granite-4.1-8b-mixed4_6 --label mixed4_6
python scripts/benchmark.py --model ./models/granite-4.1-8b-mxfp4 --label mxfp4
```

### 5. Benchmark all models at once

```bash
python scripts/benchmark.py --all
```

### 6. Generate comparison report

```bash
python scripts/generate_report.py
# → reports/final_benchmark.md
```

---

## Full Pipeline (end to end)

```bash
source /Users/sahil/venv/mlx/bin/activate
cd /Users/sahil/MLX_Tests/MLX-Quantisation

python scripts/setup_datasets.py
python scripts/quantize.py --all --mixed 4_6 --q-mode mxfp4 mxfp8 --verify
python scripts/benchmark.py --all
python scripts/generate_report.py
```

---

## Current Status

| Step | Status |
|---|---|
| Datasets downloaded | ✅ |
| FP16 model downloaded | ✅ |
| Uniform quantization (4/5/6/8 bit) | ✅ |
| Mixed-bit (mixed4_6) | ✅ |
| Block float (mxfp4, mxfp8) | 🔄 in progress |
| FP16 benchmark | ✅ |
| 4/5/6/8 bit benchmark | 🔄 in progress |
| mixed4_6 benchmark | 🔄 in progress |
| mxfp4 / mxfp8 benchmark | ⏳ pending |
| Final report | ⏳ pending |
