# Granite 4.1 8B Quantization + Benchmark Suite on Apple M5 Pro

## Why This Project

- **Model**: [IBM Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b) — recent release, Apache 2.0, enterprise-grade architecture with active IBM support
- **Opportunity**: Apple Silicon quantization benchmarks for Granite are still rare; MLX quantizations are still emerging
- **Edge**: Systematic evaluation + benchmarking is what most quantized model uploads completely lack

---

## Project Goal

Produce a rigorous quantization + benchmark suite that delivers:
1. Baseline FP16 ground truth
2. Multiple quantized variants (4bit, 5bit, 6bit, 8bit)
3. Performance benchmarks (throughput, memory, stability)
4. Quality evaluations (GSM8K, HumanEval, MMLU, long-context)
5. Published models on Hugging Face
6. Public benchmark report

---

## Directory Structure

```
MLX-Quantisation/
│
├── models/              # Quantized model outputs
├── benchmarks/          # Benchmark scripts and results
├── evals/               # Evaluation scripts
├── outputs/             # Raw generation outputs
├── reports/             # Benchmark reports and analysis
├── scripts/             # Utility scripts
├── datasets/            # Cached benchmark datasets
└── PROJECT_PLAN.md      # This file
```

---

## Phase 0 — Project Setup

**Goal**: Create folder structure and confirm environment is ready.

```bash
# Create directories
mkdir -p models benchmarks evals outputs reports scripts datasets
```

---

## Phase 1 — Environment Setup

### Step 1 — Create virtual environment

```bash
python3 -m venv granite-env
source granite-env/bin/activate
```

### Step 2 — Install core MLX stack

```bash
pip install -U mlx mlx-lm
```

### Step 3 — Install benchmark and eval tools

```bash
pip install \
  transformers \
  datasets \
  pandas \
  tqdm \
  psutil \
  matplotlib \
  lm-eval
```

### Step 4 — Optional tools

```bash
pip install jupyter wandb
```

### Step 5 — Authenticate with Hugging Face

```bash
huggingface-cli login
```

### Checklist
- [ ] Virtual environment created and activated
- [ ] MLX and mlx-lm installed (`python -c "import mlx; print(mlx.__version__)"`)
- [ ] All benchmark tools installed
- [ ] HF login confirmed

---

## Phase 2 — Baseline FP16 Testing

**Goal**: Establish ground truth before any quantization. Do NOT skip.

### Step 1 — Download and load model

```python
from mlx_lm import load

model, tokenizer = load("ibm-granite/granite-4.1-8b")
```

### Step 2 — Run inference test

```python
from mlx_lm import generate

prompt = "Explain quantization in simple terms."

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True
)

print(response)
```

### Step 3 — Record baseline in `reports/fp16_baseline.md`

| Metric | Value |
|---|---|
| Model size on disk | |
| RAM usage at idle | |
| Peak RAM during generation | |
| Tokens/sec prefill | |
| Tokens/sec decode | |
| Time to first token (TTFT) | |
| Max stable context length | |
| Response quality notes | |

### Checklist
- [ ] Model loads without error
- [ ] Inference produces coherent output
- [ ] All baseline metrics recorded in `reports/fp16_baseline.md`

---

## Phase 3 — First Quantization (MLX 4-bit)

**Goal**: Produce the first quantized variant. Best starting point — highest efficiency gain.

### Step 1 — Quantize

```bash
python -m mlx_lm.convert \
  --hf-path ibm-granite/granite-4.1-8b \
  --mlx-path ./models/granite-4.1-8b-4bit \
  -q
```

### Step 2 — Verify model loads and generates

```python
from mlx_lm import load, generate

model, tokenizer = load("./models/granite-4.1-8b-4bit")

response = generate(model, tokenizer, prompt="What are transformer models?", verbose=True)
print(response)
```

### Step 3 — Record metadata in `reports/4bit_metadata.md`

Record the following immediately after quantization:

| Field | Value |
|---|---|
| mlx-lm version | |
| macOS version | |
| Hardware | Apple M5 Pro |
| Quantization bits | 4 |
| Model size on disk | |
| Quantization time | |
| Tokenizer used | |
| Default generation params | |

### Checklist
- [ ] Quantization completes without error
- [ ] Model loads from local path
- [ ] Inference produces coherent output
- [ ] Metadata recorded

---

## Phase 4 — Performance Benchmarking

**Goal**: Measure performance characteristics across all quantization levels.

### Metrics to collect

#### Throughput
| Metric | Description |
|---|---|
| Prefill speed | Tokens/sec during prompt processing |
| Decode speed | Tokens/sec during generation |
| TTFT | Time to first token (ms) |
| Sustained throughput | Speed during long (500+ token) generations |

#### Memory
- Peak RAM (using `psutil`)
- RAM after model load
- Swap usage
- Memory under long-context load

#### Stability
- Crashes or OOM errors
- Context length failures
- Degradation during long generation runs

#### Quality (Phase 5)
- Reasoning retention
- Hallucination rate
- Formatting consistency
- Coding correctness
- Math accuracy

### Benchmark Script Skeleton — `scripts/benchmark.py`

```python
import time
import psutil
import mlx.core as mx
from mlx_lm import load, generate

def get_memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def benchmark_model(model_path, prompts, max_tokens=200):
    model, tokenizer = load(model_path)
    results = []

    for prompt in prompts:
        mem_before = get_memory_mb()
        start = time.time()

        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

        elapsed = time.time() - start
        mem_peak = get_memory_mb()

        results.append({
            "prompt": prompt,
            "response": response,
            "time_s": elapsed,
            "mem_before_mb": mem_before,
            "mem_peak_mb": mem_peak,
        })

    return results
```

### Checklist
- [ ] Benchmark script runs for FP16
- [ ] Benchmark script runs for 4bit
- [ ] All metrics captured and saved to `benchmarks/`

---

## Phase 5 — Evaluation Suite

**Goal**: Run quality evaluations on a representative subset of standard benchmarks.

### Benchmark Plan

| Benchmark | Samples | Purpose |
|---|---|---|
| GSM8K | 25 | Math reasoning |
| HumanEval | 20 | Code generation |
| MMLU | 50 | World knowledge |
| Long-context prompts | 10 | Context retention |
| Manual prompts | 20 | Qualitative eval |

### Dataset Sources

- **GSM8K**: `openai/gsm8k` on Hugging Face
- **HumanEval**: `openai/openai_humaneval` on Hugging Face
- **MMLU**: `cais/mmlu` on Hugging Face

### Loading datasets

```python
from datasets import load_dataset

gsm8k = load_dataset("openai/gsm8k", "main", split="test[:25]")
humaneval = load_dataset("openai/openai_humaneval", split="test[:20]")
mmlu = load_dataset("cais/mmlu", "all", split="test[:50]")
```

### Checklist
- [ ] GSM8K subset downloaded
- [ ] HumanEval subset downloaded
- [ ] MMLU subset downloaded
- [ ] 10 long-context prompts written to `datasets/long_context_prompts.json`
- [ ] 20 manual prompts written to `datasets/manual_prompts.json`

---

## Phase 6 — Evaluation Pipeline

**Goal**: Automate generation, scoring, and report production.

### Pipeline: `scripts/evaluate.py`

```
Load Model
   ↓
Load Benchmark Samples
   ↓
Generate Responses
   ↓
Store Outputs (outputs/<model>/<benchmark>.jsonl)
   ↓
Score Outputs
   ↓
Generate Report (reports/<model>_eval.md)
```

### Per-sample record schema

Every generation should log:

| Field | Example |
|---|---|
| model | granite-4.1-8b |
| quantization | 4bit |
| runtime | MLX |
| benchmark | GSM8K |
| prompt | "What is 12 * 15?" |
| response | "The answer is 180." |
| tokens_per_sec | 45.2 |
| ttft_ms | 210 |
| peak_ram_mb | 4820 |
| correct | Yes/No |
| notes | "hallucinated step 2" |

### Checklist
- [ ] Evaluation pipeline runs end-to-end on FP16
- [ ] Outputs stored in structured JSONL format
- [ ] Scoring function implemented for GSM8K (exact answer match)
- [ ] Scoring function implemented for MMLU (multiple choice)
- [ ] HumanEval pass@1 scoring implemented
- [ ] Per-model summary report auto-generated

---

## Phase 7 — Additional Quantization Variants

**Goal**: Produce all four quantization levels for comparison.

### Variants

| Variant | Goal |
|---|---|
| 8bit | Nearest to FP16 quality |
| 6bit | Balanced quality/efficiency |
| 5bit | Optimized |
| 4bit | Maximum efficiency (done in Phase 3) |

### Commands

```bash
# 8-bit
python -m mlx_lm.convert \
  --hf-path ibm-granite/granite-4.1-8b \
  --mlx-path ./models/granite-4.1-8b-8bit \
  -q --q-bits 8

# 6-bit
python -m mlx_lm.convert \
  --hf-path ibm-granite/granite-4.1-8b \
  --mlx-path ./models/granite-4.1-8b-6bit \
  -q --q-bits 6

# 5-bit
python -m mlx_lm.convert \
  --hf-path ibm-granite/granite-4.1-8b \
  --mlx-path ./models/granite-4.1-8b-5bit \
  -q --q-bits 5
```

### Checklist
- [ ] 8bit model produced and verified
- [ ] 6bit model produced and verified
- [ ] 5bit model produced and verified
- [ ] All variants run through Phase 6 evaluation pipeline

---

## Phase 8 — Comparison Report

**Goal**: Produce the final structured analysis in `reports/final_benchmark.md`.

### Main comparison table

| Quant | Disk Size | Peak RAM | tok/s | GSM8K | HumanEval | MMLU | Notes |
|---|---|---|---|---|---|---|---|
| FP16 | | | | | | | |
| 8bit | | | | | | | |
| 6bit | | | | | | | |
| 5bit | | | | | | | |
| 4bit | | | | | | | |

### Failure analysis table

Document where each quantization level degrades — this is what makes the report valuable:

| Quant | Failure Type | Frequency | Example |
|---|---|---|---|
| 4bit | | | |
| 5bit | | | |
| 6bit | | | |
| 8bit | | | |

### Report Sections

1. Executive Summary (2-3 paragraphs)
2. Hardware and Environment
3. Quantization Methods
4. Performance Benchmarks (tables + charts)
5. Quality Evaluations (per benchmark)
6. Failure Analysis
7. Recommendations (which quant level for which use case)
8. Raw Data Links

### Checklist
- [ ] All tables filled
- [ ] Failure analysis completed
- [ ] Charts generated (`matplotlib` — disk size, tok/s, accuracy vs quant level)
- [ ] Recommendations section written
- [ ] Report saved to `reports/final_benchmark.md`

---

## Phase 9 — Publish to Hugging Face

**Goal**: Upload quantized models with high-quality model cards.

### Naming convention

```
yourusername/granite-4.1-8b-4bit-mlx
yourusername/granite-4.1-8b-6bit-mlx
yourusername/granite-4.1-8b-8bit-mlx
```

> Build your own profile first — don't submit to mlx-community initially.

### Upload command

```bash
python -m mlx_lm.convert \
  --hf-path ibm-granite/granite-4.1-8b \
  --mlx-path ./models/granite-4.1-8b-4bit \
  -q \
  --upload-repo yourusername/granite-4.1-8b-4bit-mlx
```

### Required README sections for each repo

| Section | Required |
|---|---|
| Original model link | Yes |
| Quantization method | Yes |
| mlx-lm version used | Yes |
| Mac hardware | Yes |
| Benchmark results (table) | Yes |
| Usage example | Yes |
| Memory usage | Yes |

### Model card template

```markdown
# Granite 4.1 8B 4bit MLX

## Original Model
[ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)

## Quantization
MLX INT4 quantization via mlx-lm vX.Y.Z

## Hardware
Apple MacBook Pro, M5 Pro

## Benchmarks
| Metric | FP16 | 4bit |
|---|---|---|
| Disk size | | |
| Peak RAM | | |
| tok/s | | |
| GSM8K | | |

## Usage
\`\`\`python
from mlx_lm import load, generate
model, tokenizer = load("yourusername/granite-4.1-8b-4bit-mlx")
response = generate(model, tokenizer, prompt="Your prompt here", verbose=True)
\`\`\`
```

### Checklist
- [ ] 4bit repo created and uploaded
- [ ] 6bit repo created and uploaded
- [ ] 8bit repo created and uploaded
- [ ] Each repo has complete model card
- [ ] Each repo links back to benchmark report

---

## Phase 10 — Public Reporting

**Goal**: Build visibility and credibility from the work done.

### Publishing checklist

| Platform | Content | Status |
|---|---|---|
| Hugging Face | Quantized model repos | |
| GitHub | Scripts + benchmark code | |
| Reddit (r/LocalLLaMA) | Benchmark post with tables | |
| X/Twitter | Performance thread with key numbers | |
| Blog/Medium | Full detailed writeup | |

### What to emphasize in public posts

- Apple M5 Pro-specific measurements (rare)
- Long-context performance (most uploads skip this)
- Failure analysis and degradation patterns (almost no one publishes this)
- Quality-vs-memory tradeoff analysis
- Reasoning degradation across quantization levels

---

## Quantization Priority Order

| Priority | Method | Why |
|---|---|---|
| 1 | MLX INT4 | Maximum efficiency, most requested |
| 2 | MLX 6bit | Best quality/efficiency balance |
| 3 | MLX 8bit | Near-FP16 baseline comparison |
| 4 | AWQ | Production serving use case |
| 5 | GPTQ | Research depth, broader compatibility |

---

## Long-Term Roadmap

After completing this project, these are the natural next skills to build:

| Topic | Why It Matters |
|---|---|
| AWQ internals | Production serving |
| GPTQ math | Research depth |
| GGUF quantization | llama.cpp ecosystem |
| FP4 / MXFP4 | Future inference formats |
| Kernel optimization | Advanced infra work |
| Quantization-aware training | Frontier research |

---

## Key Insight

The real differentiator is not running `mlx_lm.convert`.

It is **systematic evaluation + benchmarking**. That is what most quantized model uploads completely lack, and it is the work that builds genuine reputation in this space.
