# Benchmark design

Why the bench scripts in `scripts/benchmarks/` look the way they do.

## What we measure

Two orthogonal things, in two separate passes:

1. **Quality** — does the quantized model still get the right answer?
   - MATH-500 (math reasoning, balanced across difficulty levels 1–5)
   - HumanEval (code, pass@1)
   - IFEval (instruction following — 11 single-rule sub-tasks)
   - long-context decode (perf only, not correctness)
2. **Performance** — measured separately under a clean steady-state regime
   (`measure_perf.py`) because the bench-suite numbers are
   thermally-throttled.

## Chat-templated prompting for thinking models

MiniCPM5-1B (and similar hybrid-thinking models) are trained with `<think>`
scaffolding in the chat template. The bench wraps every prompt in
`tokenizer.apply_chat_template(..., enable_thinking=True)`, falling back if the
template doesn't accept the kwarg.

This matters: feeding raw prompts to a thinking model produces ~zero accuracy
because the model expects the scaffold. The first version of this bench used
raw GSM8K/MMLU prompts and got near-zero on every variant — including FP16 —
which was the giveaway that the prompt format was wrong.

## Three-state scoring

`correct` / `wrong` / `no_answer`.

Standard binary scoring (correct vs everything else) lumps two very different
failure modes together:

- The model reasoned but reached the wrong answer.
- The model produced thinking-only output and never emitted a real answer.

A quantization-damaged model often collapses into the second mode — its
reasoning chain runs off the rails and no `\boxed{...}` ever appears. We want
to see that distinct from "model tried and was wrong".

When `answer_rate` (fraction of items where a parseable answer was produced)
drops below ~30%, that's a *collapse* signal, and the cards print a stronger
"Limitations" notice than for ordinary accuracy degradation.

## Stripping the thinking trace

Two cases the chat template can produce:

1. Output contains `<think>...</think>` blocks — strip them.
2. Chat template injected the opening `<think>` *into the prompt*, so model
   output starts with thinking and contains only a closing `</think>` — strip
   everything up to and including it.

Both are handled in `strip_thinking()`.

## Steady-state vs long-trace perf

The bench (`runner.py`) runs many prompts back-to-back, no cooldown, includes
long thinking traces with KV-cache growth. Decode tok/s in that regime is
*not* what a user experiences in a chat — it's the floor.

`measure_perf.py` does the user-facing measurement instead:

- warm the GPU with one prompt before measuring
- short chat-templated prompts
- `enable_thinking=False`
- report `decode_tps`, `prompt_tps`, `peak_memory_gb`

The model cards quote *both* — steady-state in the headline performance table,
long-trace in a context-scaling section — and label them explicitly.

This was a real bug-hunt: the first version of the cards reported only the
bench numbers, and quantized variants looked *slower* than FP16. The fix was
recognising that the bench was thermally-bottlenecked, not that the
quantization was broken.

## Why these benchmarks specifically

We picked the trio (MATH-500 / HumanEval / IFEval) because together they cover
the three behaviors most likely to break under quantization:

- **MATH-500** — long reasoning chains, drift compounds — breaks first.
- **HumanEval** — short, structured, format-sensitive — surprisingly robust.
- **IFEval** — rule-following — sensitive to logit drift on rare tokens.

If a quant scores ≥90% of FP16 on all three, it's safe to ship. If MATH drops
but the other two hold, that's a known and shippable tradeoff (we mark it in
the card limitations section).
