"""
Push quantized models to HuggingFace Hub.

Uses HF_TOKEN from env. Creates repos if missing, otherwise overwrites.

Usage:
  python scripts/push_to_hf.py
  python scripts/push_to_hf.py --skip 4bit 5bit          # skip these variants
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

AUTHOR     = "SahilChachra"
BASE_NAME  = "granite-4.1-8b"
VARIANTS   = ["4bit", "5bit", "6bit", "8bit", "mixed4_6", "mxfp4", "mxfp8"]


def push_one(api, variant):
    model_dir = MODELS_DIR / f"{BASE_NAME}-{variant}"
    repo_id   = f"{AUTHOR}/{BASE_NAME}-{variant}-mlx"

    if not model_dir.exists():
        return {"variant": variant, "status": "skipped", "reason": "model folder not found"}
    if not (model_dir / "README.md").exists():
        return {"variant": variant, "status": "skipped", "reason": "README.md (model card) not generated"}

    print(f"\n=== {variant} → {repo_id} ===")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"  Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        return {"variant": variant, "status": "failed", "reason": f"create_repo: {e}"}

    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {variant} MLX quantization",
        )
        print(f"  Uploaded successfully.")
        return {"variant": variant, "status": "ok", "repo_id": repo_id}
    except Exception as e:
        return {"variant": variant, "status": "failed", "reason": f"upload: {str(e)[:200]}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Variants to skip")
    parser.add_argument("--only", nargs="*", help="Run only these variants")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who['name']}")
    print(f"Token role: {who.get('auth', {}).get('accessToken', {}).get('role', 'N/A')}")

    targets = args.only if args.only else [v for v in VARIANTS if v not in args.skip]

    results = []
    for v in targets:
        results.append(push_one(api, v))

    # Summary
    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    for r in results:
        marker = "✓" if r["status"] == "ok" else ("⊘" if r["status"] == "skipped" else "✗")
        line = f"  {marker} {r['variant']}: {r['status']}"
        if r["status"] == "ok":
            line += f" → {r['repo_id']}"
        elif r.get("reason"):
            line += f" — {r['reason']}"
        print(line)

    return results


if __name__ == "__main__":
    main()
