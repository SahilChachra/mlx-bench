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
| `MLX_BENCH_SCRIPT`       | `scripts/benchmarks/runner.py` | `run_all_isolated.sh`, `run_pipeline.sh` |
| `MLX_BENCH_HF_AUTHOR`    | `sahilchachra` | cards.py, push.py |
| `MLX_BENCH_LICENSE`      | `apache-2.0` | cards.py (frontmatter) |
| `HF_TOKEN`               | — | push.py |

---

## License

Apache 2.0
