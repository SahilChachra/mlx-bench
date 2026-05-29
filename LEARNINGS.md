# Learnings — MLX Quantization Journey

Notes collected across quantizing real models for Apple Silicon: **Granite 4.1
8B**, **Hy-MT2 (7B and 1.8B)**, **MiniCPM5 1B**, and **LFM2.5-8B-A1B**. The
pipeline (`mlx-bench`) was built up incrementally; most things in this doc are
mistakes we made on one model and fixed before the next.

---

## 1. Quantization methods — what actually works when

### Affine integer (4/5/6/8-bit, group-64)

- The **default safe path**. `mlx_lm convert` does the right thing on every
  arch we tried (Llama-family, Granite, Hy-MT2, MiniCPM5, LFM2.5).
- 8-bit is effectively lossless on all 4 model families. 6-bit was almost
  always within 1pp of FP16.
- 4-bit cliff is **model-dependent**:
  - **Translation (Hy-MT2)**: tolerable; chrF++ drop ~2-4 pts.
  - **Reasoning (MiniCPM5)**: noticeable math degradation.
  - **MoE hybrid (LFM2.5 mxfp4)**: −13pp on MATH-500 but **0pp** on HumanEval
    and IFEval. Reasoning chains break first; structured/instruction tasks
    survive.
- **Mixed-bit recipes** (e.g. `mixed4_6`): marginal gains over uniform 4-bit
  on dense models — not worth the disk overhead in our experience. OptiQ
  obsoletes this when it works.

### MX FP4 / MX FP8 (microscaling block-float)

- Same disk footprint as affine 4/8-bit, slightly better dynamic range.
- Quality usually matches affine 8-bit; speed comparable.
- On LFM2.5: **mxfp4 was the speed champion** — 2.7× FP16 decode, fits
  everything (including the conv backbone) into 4-bit storage. mxfp4 is the
  best "small + fast" choice on hybrid-arch models.

### DWQ (Distillation-aware Weight Quantization)

- Helps at **low bits only** (≤4). We tested on MiniCPM5 4-bit and 8-bit:
  - 4-bit: small but real recovery vs naive affine.
  - **8-bit: silently aborts**. "Final validation loss is worse than initial."
    The naive 8-bit already matches the FP16 KL noise floor (~0.001); there's
    nothing to recover.
- Conclusion: don't bother running DWQ above 4-bit on small (<2B) models.

### OptiQ (calibration-driven per-layer mixed precision)

- The interesting one. Runs a calibration set through the FP16 reference and
  trial quantizations of each layer, ranks layers by KL drift, allocates bits
  to hit a target bpw.
- **On MiniCPM5-1B**:
  - 4.0 bpw **collapsed** (MATH-500 → 0%, no real answers). 1B-class models
    don't have enough redundancy to safely descend that low.
  - 5.0 bpw was **publishable** (MATH 36.7% vs 70% FP16, code/IFEval held).
- **On LFM2.5-8B-A1B (8B MoE)**:
  - 5.0 bpw **matched or beat FP16 on every quality benchmark**. MATH-500
    70% (= FP16, answered 26/30 vs 25/30), HumanEval 76.7% vs 73.3%, IFEval
    84.1% vs 79.5%.
- Two things drive whether OptiQ wins:
  1. **Model capacity** — bigger models tolerate lower bpw.
  2. **Architecture** — OptiQ only quantizes attention/MLP weights. Conv,
     SSM, embedding layers stay at original precision.

---

## 2. Hybrid architectures change everything (LFM2.5 lesson)

LFM2.5-8B-A1B is **18 LIV-conv layers + 6 GQA layers**. Two non-obvious
consequences:

- **OptiQ only saw 82 components** (the 6 attn/MLP layers' weights). The
  optimizer reports 5.04 bpw — that's the average over *quantized* weights.
  The on-disk model came out at **8.37 bpw overall** because the conv backbone
  is still FP16. Disk size: 8.3 GB instead of the ~2.6 GB you'd get if it
  were a fully-dense 8B at 5 bpw.
- **Context scaling is flat**. Decode at 128 vs 1024 tokens differs by <1%.
  KV-cache growth only matters for the 6 attention layers; the conv layers
  have no KV state. On MiniCPM5 (24 GQA layers) the same scan showed clear
  KV-driven slowdown beyond 512 tokens.

Always check the layer breakdown before promising "X bpw = Y MB on disk."

## 3. Where MoE gates sit in the sensitivity order

Independent finding from LFM2.5: OptiQ pushed **every MoE feed-forward gate up
to the top tier (8-bit)**. The pattern dominates over the usual "first +
last + lm_head" preserve-list seen on dense models.

Why this makes sense: gate weights pick which experts run for each token.
Small drift changes routing decisions, which compounds through every
subsequent layer. So OptiQ pays the bits to keep gates precise and compresses
the much-larger expert MLPs more aggressively.

For dense models with `lm_head`, first and last transformer blocks tend to be
the most sensitive layers (MiniCPM5 pattern). For MoE models, gates lead.
This needs to be in the card generator's narrative — and it is, now.

---

## 4. Benchmark design — what we actually want to measure

### Chat templates + thinking mode

The hardest bug we hit was on MiniCPM5: raw-prompt GSM8K scored near-zero on
**every variant including FP16**. That's the giveaway — the prompt format was
wrong, not the quantization.

Fix: wrap every prompt through `tokenizer.apply_chat_template(...,
add_generation_prompt=True, enable_thinking=True)`. Modern hybrid-thinking
models (MiniCPM5, LFM2.5) need the `<think>` scaffolding the template
injects. Without it, the model produces empty output or repetitive garbage.

Per-benchmark choice:
- **MATH-500, HumanEval, long-context** → `enable_thinking=True` (CoT helps).
- **IFEval** → `enable_thinking=False`. Instruction-following tasks need a
  direct answer. A reasoning chain just burns tokens and sometimes leaks into
  the response, failing structural rules like "no commas" or "all
  lowercase."

### Three-state scoring

`correct / wrong / no_answer`.

A quantization-damaged model often **collapses** — it emits a `<think>` block
that runs off the rails and never produces a real answer. That's
distinguishable from "tried and got it wrong." Binary scoring lumps both into
"incorrect" and hides the failure mode.

When `answer_rate` < 30%, that's a hard collapse signal (e.g. OptiQ-4bpw on
MiniCPM5). When `answer_rate` ≈ FP16's but accuracy drops, that's the model
trying its best with degraded weights — a more honest tradeoff.

### Stripping the thinking trace

Two cases the chat template produces:
1. Output contains `<think>...</think>` — strip the block.
2. Chat template **injected the opening `<think>` into the prompt**, so the
   model's output starts with thinking and contains **only the closing
   `</think>`**. Strip everything up to and including it.

Hit case 2 on MiniCPM5; case 1 on LFM2.5. Handle both regexes or some models
will score zero for parser reasons.

### max_tokens

A reasoning model can spend several thousand tokens just thinking before
emitting the answer. If `max_tokens` is too tight, the trace truncates
mid-think and we record `no_answer` — falsely undercounting.

What we settled on for thinking models:
- **MATH-500**: 8192 (level-4/5 problems can need 4k+ thinking tokens).
- **HumanEval**: 4096 (CoT + code block fits well under this).
- **IFEval**: 2048 (thinking disabled, but the response can still be long).
- **AIME**: 16384 (we mostly dropped AIME — its problems need too many
  tokens to fit a reasonable bench budget).

Started MiniCPM5 at 4096/1024 for MATH/IFEval and got phantom `no_answer`
rates. Bumped for LFM2.5; the answer-rate jumped on both.

---

## 5. Perf measurement — bench numbers are not what users see

The single biggest reporting bug we shipped and then fixed: the first version
of model cards showed **only** the benchmark's average decode tok/s. Quantized
variants looked *slower* than FP16. The user-facing experience said otherwise.

Two effects compound:

- **Thermal throttling.** Quantized models are more compute-bound; FP16 is
  memory-bound. Under sustained back-to-back generation (no cooldown), quants
  throttle harder. On MiniCPM5: bench showed 8bit ~88 t/s, FP16 ~144 t/s.
  Clean steady-state measurement: 8bit 244 t/s, FP16 144 t/s.
- **KV-cache growth.** On reasoning models with long traces, decode drops as
  the cache grows. Bench averages over the whole trace; chat users mostly see
  the first few hundred tokens.

So we measure perf two ways now:

- **Steady-state** (`measure_perf.py`): warm with one prompt, then 3 short
  prompts, chat-templated, `enable_thinking=False`, peak memory. This is the
  number that goes in the card's headline performance table.
- **Long-trace average**: from the regular bench. This is the floor — labeled
  as such.

Both numbers are in the card. The header table quotes steady-state with the
caveat: *"Warmed, short-prompt, chat-templated, thinking disabled. Long
thinking traces will be slower due to KV-cache growth."*

---

## 6. Calibration mixes and OptiQ

mlx-optiq v0.1.0 shipped without its bundled `optiq.jsonl` calibration mix.
We built our own (`benchmarks/build_optiq_calibration.py`): 32 samples across
4 domains — prose (wikitext), thought (MATH-500 problems), code (HumanEval),
constraint (IFEval). Same shape as the upstream schema.

v0.2+ ships the official 6-domain mix and adds tool-call + agent-loop
domains. Both work; the 4-domain mix is fine for general chat models and
worked for both MiniCPM5 and LFM2.5.

CLI changed between versions — note for future:
- v0.1: `optiq convert --model X --output Y --calibration path.jsonl`
- v0.2+: `optiq convert X --output Y --calibration-mix path.jsonl`

The wrapper at `scripts/quantization/optiq.py` was updated mid-LFM2.5 run.

---

## 7. Process / tooling gotchas

- **HuggingFace XET upload stalls.** During the LFM2.5 push, `huggingface_hub
  upload_folder` hung at 99.7% for 3+ hours after the bytes had uploaded —
  the commit step never fired. Killing and retrying with `hf upload` (the
  CLI) finished in ~1 minute (XET dedup vs the prior failed CAS objects).
  **Prefer the CLI for any upload above ~5 GB.**
- **mlx-optiq needs `torch`** — but only because it uses `safetensors
  safe_open(framework="pt")` to read the FP16 base. Not documented; `uv pip
  install torch` fixes the `ModuleNotFoundError`.
- **`run_all_isolated.sh` matters more than it looks.** Running benchmarks
  back-to-back in one Python process **double-counts peak memory** (mlx
  reports process-wide). One process per variant + 2 min cooldown is the
  baseline for honest comparisons.
- **MoE models load slowly.** LFM2.5 takes ~30s to load — `mlx-lm` reads
  every expert. Factor this into per-variant bench budgets.
- **Model size budget on M5 Pro (24 GB unified)**: FP16 of 8B fits easily.
  FP16 of 13B+ is borderline — peak memory during inference can exceed unified
  RAM and swap kicks in. Plan accordingly.

---

## 8. Things we tried that didn't work

- **DWQ at 8-bit** — silently aborted (see §1).
- **OptiQ at 4.0 bpw on MiniCPM5-1B** — collapsed (MATH-500 → 0%). Published
  as a finding tweet, not as a model.
- **AIME as a default benchmark** — problems require 8k–16k tokens of CoT;
  total bench time blew up. Dropped in favor of MATH-500 sampled across
  levels, which gives a similar quality signal in ~10× less wall time.
- **GSM8K + MMLU on a reasoning model** — every variant scored ~0%. The
  prompts weren't chat-templated. After fixing the prompting we still moved
  to MATH-500 + IFEval because they're closer to what these models are
  actually trained for.
- **Pretty `mlx_lm convert --verify` output** — verify uses a *raw* prompt
  (no chat template), so even a healthy quantized chat model produces
  repetitive nonsense. Not a quantization problem; always re-verify with the
  chat template before declaring a quant broken.

---

## 9. The one-line guide

| Want… | Pick |
|---|---|
| Lossless, easy | `8bit` or `mxfp8` |
| ~Half disk, ~lossless | `6bit` or `optiq-5bpw` (if attn/MLP is most of the model) |
| Smallest viable | `4bit` / `mxfp4` — accept a math/reasoning hit |
| Maximum decode tok/s | `mxfp4` — wins on every model we tested |
| Lowest quality drop at low bits | `optiq-5bpw` (or `optiq-4bpw` only on ≥7B models) |
| Translation models | `mxfp8` — `mxfp4`/4-bit chrF++ drops are noticeable |

---

## 10. Open questions

- **OptiQ at 3.5 bpw** on 8B+ models — unexplored. Quality might still hold.
- **Per-expert quantization for MoE** — current pipeline quantizes all
  experts at the same tier. Could KL-rank experts individually for further
  compression?
- **Quant-aware sampling parameters** — we use greedy decode everywhere for
  determinism. LiquidAI recommends temp 0.2 + top_p 80 for LFM2.5; their
  numbers (MATH-500 88.76) come from that sampling. Quant deltas under
  sampled decoding might differ from greedy.
- **DWQ + OptiQ stacked** — apply OptiQ to set bit-tiers, then DWQ-tune the
  low-bit tiers. Not tried yet.
