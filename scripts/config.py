"""
Shared config for the mlx-bench pipeline.

Resolves the model under test from environment variables so the same scripts
can be reused for different base models without code edits.

Set in your shell before running any script:

    export MLX_BENCH_BASE_NAME="hy-mt2-7b"           # local dir prefix
    export MLX_BENCH_HF_REPO="tencent/Hy-MT2-7B"     # source HF repo
    export MLX_BENCH_DISPLAY_NAME="Hy-MT2-7B"        # name used in cards/text

Defaults keep the original Granite pipeline working.
"""

import os

BASE_NAME    = os.environ.get("MLX_BENCH_BASE_NAME", "granite-4.1-8b")
BASE_HF_REPO = os.environ.get("MLX_BENCH_HF_REPO",   "ibm-granite/granite-4.1-8b")
DISPLAY_NAME = os.environ.get("MLX_BENCH_DISPLAY_NAME", BASE_NAME)
