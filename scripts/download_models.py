"""Bootstrap the model files needed by the desktop sidecar (and the prod
server) on a fresh clone.

The ~180 MB ONNX checkpoint and 240 kB tokenizer are too big / wrong-shaped
for git, so they live on Hugging Face at:

    https://huggingface.co/karansea/indicconformer-stt-pa-ctc-shabad-preview

This script pulls them down with `huggingface_hub.snapshot_download` (which
uses the HF cache, dedups across calls, and supports resumable / xet
downloads), then materializes the names the rest of the codebase expects:

    models/shabad-id-models/v4.int8.onnx       (HF: model.int8.onnx)
    models/shabad-id-models/v4_tokenizer.model (HF: tokenizer.model)

Usage:

    pip install huggingface_hub
    python scripts/download_models.py            # idempotent — skip if up to date
    python scripts/download_models.py --force    # re-link even if files exist

Designed to be safe to run from anywhere (cwd-independent), idempotent
(skips files already in place), and cross-platform (uses `Path.symlink_to`
which raises gracefully on Windows without dev mode + falls back to copy).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "karansea/indicconformer-stt-pa-ctc-shabad-preview"

# (HF filename, local filename inside models/shabad-id-models/)
# Keep this list short — the desktop sidecar only needs the int8 model.
# The fp32 model.onnx is 480 MB and only used by the training/eval path,
# so we don't pull it by default.
FILES = [
    ("model.int8.onnx", "v4.int8.onnx"),
    ("tokenizer.model", "v4_tokenizer.model"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models" / "shabad-id-models"

_MIN_VALID_BYTES = 1024  # detect symlink-as-text stubs (~45 bytes)


def _link_or_copy(src: Path, dst: Path) -> str:
    """Materialize `dst` from `src`. Returns 'symlink' or 'copy' for logging.

    Prefers a relative symlink (cheap, lets HF's cache stay the source of
    truth so a `huggingface-cli cache prune` can clean up). Falls back to
    `shutil.copy2` on Windows where unprivileged users can't symlink unless
    Developer Mode is on — paying the 184 MB once is fine.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        # Relative symlink so the tree is portable if the user moves the
        # whole `bani` checkout (the HF cache lives outside the repo).
        # On Windows, this can also raise ValueError when src and dst live
        # on different drives (e.g. CI: HF cache on C:, repo on D:) — no
        # cross-drive relative path is expressible there. Fall back to copy.
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
        return "symlink"
    except (OSError, NotImplementedError, ValueError):
        # Windows without Developer Mode, sandboxed FS, OR cross-drive on
        # Windows (e.g. GitHub Actions runners). Copy is always safe.
        shutil.copy2(src, dst)
        return "copy"


def _is_up_to_date(dst: Path, expected_size: int) -> bool:
    """Skip the download dance if the file we'd produce already matches."""
    if not dst.exists():
        return False
    try:
        return dst.stat().st_size == expected_size
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download + re-link even if files appear up to date.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"HuggingFace repo to pull from (default: {REPO_ID}).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print(
            "[download_models] huggingface_hub not installed.\n"
            "    pip install huggingface_hub\n"
            "    (or `pip install -r desktop/sidecar/requirements-bootstrap.txt`)",
            file=sys.stderr,
        )
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: if every destination already exists with the right size,
    # nothing to do. snapshot_download is itself fast on a warm cache, but
    # this lets us bypass the HF API call entirely on every Tauri build.
    if not args.force:
        all_present = all(
            (MODELS_DIR / local).exists()
            and (MODELS_DIR / local).stat().st_size >= _MIN_VALID_BYTES
            for _, local in FILES
        )
        if all_present:
            print("[download_models] all files present, nothing to do.")
            print(f"    {MODELS_DIR}")
            for _, local in FILES:
                p = MODELS_DIR / local
                print(f"      {local:24s} {p.stat().st_size:>13,d} bytes")
            return 0

    print(f"[download_models] pulling {args.repo_id} ...")
    cache_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            allow_patterns=[hf for hf, _ in FILES],
        )
    )
    print(f"    cache: {cache_path}")

    for hf_name, local_name in FILES:
        src = cache_path / hf_name
        dst = MODELS_DIR / local_name
        if not src.exists():
            print(f"  ✗ {hf_name} missing in HF snapshot — repo layout changed?",
                  file=sys.stderr)
            return 1
        if not args.force and _is_up_to_date(dst, src.stat().st_size):
            print(f"  ✓ {local_name:24s} up to date")
        else:
            method = _link_or_copy(src, dst)
            print(f"  + {local_name:24s} ({method}) <- {hf_name}")

    print(
        f"\n[download_models] done. "
        f"Model at: {MODELS_DIR / 'v4.int8.onnx'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
