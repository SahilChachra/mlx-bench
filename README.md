# mlx-bench

A reproducible quantization + benchmarking pipeline for **MLX LLMs** on Apple Silicon.

Quantize any HuggingFace LLM that [mlx-lm](https://github.com/ml-explore/mlx-lm)
supports into every available variant (affine 4/5/6/8-bit, mixed-bit, MX FP4/FP8,
[DWQ](https://github.com/ml-explore/mlx-lm), and [OptiQ](https://mlx-optiq.com/)
mixed-precision), measure performance **and** quality side-by-side, and publish
the result with auto-generated model cards.

**Runtime**: MLX + mlx-lm
**Hardware**: Any Apple Silicon Mac (M1 or later)

---

## Docs

- [`docs/QUANTIZATION_METHODS.md`](docs/QUANTIZATION_METHODS.md) — what each method does, when to use it, what we learned.
- [`docs/BENCHMARK_DESIGN.md`](docs/BENCHMARK_DESIGN.md) — why the bench uses chat templates, 3-state scoring, separate steady-state perf, etc.
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — end-to-end walkthrough: env vars, commands, output layout.

---

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install mlx-lm datasets huggingface_hub tqdm sacrebleu mlx-optiq

export MLX_BENCH_BASE_NAME="minicpm5-1b"
export MLX_BENCH_HF_REPO="openbmb/MiniCPM5-1B"
export MLX_BENCH_DISPLAY_NAME="MiniCPM5-1B"

hf download "$MLX_BENCH_HF_REPO" --local-dir "models/${MLX_BENCH_BASE_NAME}-fp16"
python -m scripts.benchmarks.setup_datasets thinking

# quantize → bench → card → report (+ optional --push)
python -m scripts.pipeline all --method affine --push
```

For OptiQ:

```bash
python -m scripts.benchmarks.build_optiq_calibration
python -m scripts.pipeline quantize --method optiq --target-bpw 5.0 --candidate-bits 3,4,6,8
python -m scripts.benchmarks.runner --model "models/${MLX_BENCH_BASE_NAME}-optiq-5bpw" --label optiq-5bpw
python -m scripts.pipeline card
HF_TOKEN=... python -m scripts.pipeline push --only optiq-5bpw
```

### Vision-language models (VLMs)

The same pipeline drives mlx-vlm when `MLX_BENCH_MODALITY=vlm`. mlx-vlm
supports affine 4/6/8-bit, mxfp4, mxfp8, and mixed recipes (`mixed_4_6`,
`mixed_3_6`, etc.) — but **not** DWQ or OptiQ.

VLM workflows live in a **separate venv** (`requirements-vlm.txt`) because
mlx-vlm pulls in fastapi/torchvision/pydantic that don't belong in the
LLM venv, and some VLM model branches require older `transformers`
versions that conflict with the mlx-lm pins.

```bash
uv venv ~/venv/mlx-vlm --python 3.12
source ~/venv/mlx-vlm/bin/activate
uv pip install -r requirements-vlm.txt

# Example: nvidia/LocateAnything-3B — substitute any mlx-vlm-supported repo
export MLX_BENCH_MODALITY=vlm
export MLX_BENCH_BASE_NAME=locateanything-3b
export MLX_BENCH_HF_REPO=nvidia/LocateAnything-3B
export MLX_BENCH_DISPLAY_NAME="LocateAnything 3B"

# Quantize directly from HF (no local FP16 download needed for VLM)
python -m scripts.pipeline quantize --method affine --q-mode mxfp4

# Text benchmarks run on the LM tower (image=None passed internally)
python -m scripts.benchmarks.runner --model "models/${MLX_BENCH_BASE_NAME}-mxfp4" --label mxfp4

# Grounding benchmark (RefCOCOg val). Flags below are tuned for LocateAnything;
# adjust --prompt / --generation-mode for other VLMs.
python -m scripts.benchmarks.refcoco \
    --model "models/${MLX_BENCH_BASE_NAME}-mxfp4" --label mxfp4 --n 200 \
    --register-locateanything
```

The text suites (HumanEval/IFEval/MATH-500/MMLU/long-context) all drive
the LM tower text-only. `refcoco.py` is the first grounding-aware suite;
its defaults match NVIDIA's recommendation for LocateAnything but
`--prompt` / `--generation-mode` are exposed for other VLMs.

---

## Repo layout

```
.
├── docs/
│   ├── QUANTIZATION_METHODS.md
│   ├── BENCHMARK_DESIGN.md
│   └── PIPELINE.md
│
├── scripts/
│   ├── pipeline.py                   # unified CLI: `python -m scripts.pipeline ...`
│   ├── config.py                     # reads MLX_BENCH_* env vars
│   │
│   ├── quantization/
│   │   ├── affine.py                 # affine 4/5/6/8 + mixed + MX FP
│   │   ├── dwq.py                    # mlx-lm distillation-aware
│   │   └── optiq.py                  # mlx-optiq per-layer mixed precision
│   │
│   ├── benchmarks/
│   │   ├── runner.py                 # MATH-500 / HumanEval / IFEval / long-ctx
│   │   ├── flores.py                 # translation (chrF++ / BLEU)
│   │   ├── measure_perf.py           # steady-state perf only
│   │   ├── setup_datasets.py         # download / sample all benchmark data
│   │   └── build_optiq_calibration.py
│   │
│   ├── publish/
│   │   ├── cards.py                  # per-variant HF model card
│   │   ├── report.py                 # cross-variant comparison report
│   │   └── push.py                   # push variants to HF
│   │
│   ├── run_pipeline.sh               # legacy bash entry point
│   └── run_all_isolated.sh           # bench every variant in isolated subprocesses
│
├── models/                           # (gitignored) FP16 + quantized variants
├── datasets/                         # (gitignored except small jsonls)
├── outputs/                          # (gitignored) per-variant summary.json
└── reports/                          # (gitignored) cross-variant markdown reports
```

---

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `MLX_BENCH_BASE_NAME`    | `granite-4.1-8b` | All scripts (folder + repo naming) |
| `MLX_BENCH_HF_REPO`      | `ibm-granite/granite-4.1-8b` | quantize, report, cards |
| `MLX_BENCH_DISPLAY_NAME` | `$BASE_NAME` | report / card title |
| `MLX_BENCH_MODALITY`     | `llm` | `llm` or `vlm` — routes loader/quantize through mlx-lm or mlx-vlm |
| `MLX_BENCH_SCRIPT`       | `scripts/benchmarks/runner.py` | `run_all_isolated.sh`, `run_pipeline.sh` |
| `MLX_BENCH_HF_AUTHOR`    | `sahilchachra` | cards.py, push.py |
| `MLX_BENCH_LICENSE`      | `apache-2.0` | cards.py (frontmatter) |
| `HF_TOKEN`               | — | push.py |

---

## License

Apache 2.0
