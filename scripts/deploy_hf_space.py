"""Deploy the app to a Hugging Face Space (Docker SDK, free CPU tier).

Builds a minimal bundle (code + artifacts + Dockerfile + Space README) and
uploads it with the Hub API, which handles binary files (LFS/Xet) automatically.

Usage:
    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py              # deploy
    python scripts/deploy_hf_space.py --dry-run                     # build bundle only

The Space id defaults to "<token owner>/forest-cover-type-classifier";
override with --space-id or the HF_SPACE_ID environment variable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPACE_NAME = "forest-cover-type-classifier"
GITHUB_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/evez12/forest-cover-type-classifier")

BUNDLE_DIRS: tuple[str, ...] = ("covtype", "backend", "frontend", "artifacts")
BUNDLE_FILES: tuple[str, ...] = ("Dockerfile", "requirements-api.txt", "LICENSE")

SPACE_README = f"""---
title: Forest Cover Type Classifier
emoji: 🌲
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
license: mit
short_description: PyTorch MLP predicting forest cover type from terrain data
tags:
  - pytorch
  - fastapi
  - tabular-classification
---

# 🌲 Forest Cover Type Classifier

PyTorch MLP trained on the UCI Covertype dataset (54 cartographic features → 7 cover types),
served with FastAPI. Test accuracy **95.4 %**, macro F1 **0.922**.

- **Web UI:** this Space
- **API docs:** append `/docs` to the Space app URL
- **Source code:** [{GITHUB_URL}]({GITHUB_URL})

This Space is deployed automatically from GitHub by CI — do not edit files here.
"""


def build_bundle(dest: Path) -> None:
    for name in BUNDLE_DIRS:
        shutil.copytree(
            PROJECT_ROOT / name,
            dest / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in BUNDLE_FILES:
        shutil.copy2(PROJECT_ROOT / name, dest / name)
    (dest / "README.md").write_text(SPACE_README, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space-id", default=os.getenv("HF_SPACE_ID"), help="<user>/<space-name>")
    parser.add_argument("--dry-run", action="store_true", help="Build the bundle and list it, no upload")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        build_bundle(bundle)
        files = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file())
        print(f"Bundle: {len(files)} files")
        for f in files:
            print(f"  - {f}")
        if args.dry_run:
            return

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN environment variable is required")
        api = HfApi(token=token)
        space_id = args.space_id or f"{api.whoami()['name']}/{DEFAULT_SPACE_NAME}"

        api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=bundle,
            commit_message=f"Deploy {os.getenv('GITHUB_SHA', 'local build')[:7]} from GitHub",
            delete_patterns="*",  # mirror the bundle exactly (removes stale files)
        )
        user, name = space_id.split("/")
        print(f"\nDeployed: https://huggingface.co/spaces/{space_id}")
        print(f"App URL : https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
