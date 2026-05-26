# Quantization methods in mlx-bench

A quick reference for the quantization methods this pipeline can produce, what
they actually do, when to reach for them, and what we learned running them on
real models on Apple Silicon (M5 Pro).

All numbers below come from runs on `openbmb/MiniCPM5-1B` (a small hybrid-thinking
model). They are illustrative — the qualitative shape generalizes; absolute
numbers will differ by base model.

---

## 1. Affine integer quantization — `quantization/affine.py`

Standard post-training quantization. Every weight in a group is mapped to an
`n`-bit integer with a per-group `(scale, zero_point)` pair:

    w_q = round((w - zero) / scale)

- `--bits 4 | 5 | 6 | 8` — uniform across all quantizable layers.
- `--mixed 4_6` — group_size-aligned mixed: 4-bit on most, 6-bit on outliers.
- `--q-mode mxfp4 mxfp8` — MX FP4 / FP8 microscaling formats (E2M1 / E4M3 with
  shared block scale). Better dynamic range than affine int at the same bits.

**When to use**

- `8bit` / `mxfp8`: near-lossless for most workloads. Default safe choice.
- `6bit` / `mxfp4`: ~half the size of FP16, minimal quality loss on most models.
- `4bit`: ~4× smaller, robust on knowledge / code / chat, can degrade reasoning.
- `mixed_4_6`: marginal gains vs uniform 4-bit on a few sensitive layers.

**Practical notes**

- On Apple Silicon, lower-bit quants are more compute-bound — under sustained
  load they thermally throttle harder than FP16, which is memory-bound. The
  bench numbers (long traces, no cooldown) will look slower than the warmed
  steady-state numbers we publish in cards.
- Group size 64 is a reasonable default; smaller groups = more scales = bigger
  files; larger groups = quality cliff on outlier-heavy layers.

---

## 2. DWQ — Distillation-aware Weight Quantization — `quantization/dwq.py`

Wraps `mlx_lm.quant.dwq`. After an initial affine quant, DWQ tunes the
per-group `(scale, zero)` (~0.007% of params are trainable) against the FP16
teacher by minimizing KL divergence on a calibration set.

**When to use**

- **Low bits only** — DWQ has room to recover quality at 4-bit and below.
- Worth running when you've seen real degradation at 4-bit and don't want to
  bump to 5/6-bit for the disk savings.

**What we learned**

- At 8-bit DWQ silently aborts on small models: the naive 8-bit already hits
  the FP16 KL noise floor (~0.001). There's nothing to recover.
- DWQ runs for ~30 min to a few hours on an M-series chip depending on model
  size and `--num-samples`.

---

## 3. OptiQ — per-layer KL-sensitivity mixed precision — `quantization/optiq.py`

Wraps the [mlx-optiq](https://mlx-optiq.com/) `optiq convert` CLI. Runs a
calibration set through the FP16 reference and trial quantizations of each
layer, measures per-layer output drift, and assigns bits unequally so the
*average* bpw hits a target while the *most sensitive* layers stay high.

Pattern observed in practice: `lm_head`, the **first**, and the **last**
transformer block are typically pushed up to the high tier (8-bit); the middle
gets aggressively compressed.

**When to use**

- When you want to target a specific bits-per-weight (e.g. 5.0) and let the
  model decide where to spend them.
- When uniform low-bit affine collapses but you don't want to go all the way
  back to 6-bit / 8-bit on every layer.

**What we learned**

- **4.0 bpw collapsed on MiniCPM5-1B** — MATH-500 dropped to 0% (no real
  answers; the model emitted thinking-scaffold only). 1B-class models seem to
  not have enough redundancy to safely descend that low even with per-layer
  routing.
- **5.0 bpw was publishable** — MATH-500 36.7% vs 70% FP16 (degraded but
  real), HumanEval 67%, IFEval 68%. Decode 329 t/s steady-state vs 144 for
  FP16. A genuine sweet spot for chat/code workloads where heavy reasoning is
  not the primary use case.
- The PyPI v0.1.0 package ships without its bundled `optiq.jsonl` calibration
  file — `benchmarks/build_optiq_calibration.py` builds a 32-sample 4-domain
  replacement (prose + multi-step thought + code + constraint-following) from
  public datasets.
- OptiQ writes a nested layout: `<out>/optiq_mixed/` plus sibling baselines
  (`static_mixed/`, `uniform_4bit/`). `optiq.py` promotes `optiq_mixed/` to
  the parent directory and prunes the others so the rest of the pipeline sees
  a flat variant folder.

---

## Picking a recipe

|Workload | Recommended |
|---|---|
| Heavy math / multi-step reasoning | FP16, or `8bit` / `mxfp8` |
| General chat + code + instruction-following | `8bit` (safe), `6bit` (~2× smaller), or `optiq-5bpw` (~2.5× smaller, fast decode) |
| Aggressive size, tolerant of some quality drop | `4bit` / `mxfp4` / `optiq-5bpw` |
| Below 4 bpw | Likely not viable on ≤1B models — measure before publishing |
