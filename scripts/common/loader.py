"""
Modality-aware model loading + text generation.

Both mlx-lm and mlx-vlm expose `load()` and `stream_generate()` with
slightly different shapes:

  mlx-lm  : load(path) -> (model, tokenizer)
            stream_generate(model, tokenizer, prompt=..., max_tokens=...)

  mlx-vlm : load(path) -> (model, processor)
            stream_generate(model, processor, prompt=..., image=..., max_tokens=...)

This shim returns a uniform `(model, tok)` from `load_model()` where `tok`
exposes `.apply_chat_template(...)` in both cases, so all existing
text-only benchmarks keep working unchanged. `text_stream_generate()`
dispatches to the right backend internally.

VLM models still accept text-only prompts — we pass `image=None` so the
LM tower runs alone. That's enough for our text reasoning/coding/IFEval
suites.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODALITY


class _VLMTokAdapter:
    """Lets benchmark code keep calling `tok.apply_chat_template(...)` while
    holding the underlying mlx-vlm processor for stream_generate."""

    def __init__(self, processor):
        self._processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._processor, name)


def load_model(path):
    if MODALITY == "vlm":
        from mlx_vlm import load as vlm_load
        model, processor = vlm_load(path)
        return model, _VLMTokAdapter(processor)
    from mlx_lm import load as lm_load
    return lm_load(path)


def text_stream_generate(model, tok, prompt, **kwargs):
    """Streaming text generation that works for both LLMs and VLMs."""
    if MODALITY == "vlm":
        from mlx_vlm import stream_generate as vlm_stream
        processor = tok._processor if isinstance(tok, _VLMTokAdapter) else tok
        yield from vlm_stream(model, processor, prompt=prompt, image=None, **kwargs)
        return
    from mlx_lm import stream_generate as lm_stream
    yield from lm_stream(model, tok, prompt=prompt, **kwargs)
