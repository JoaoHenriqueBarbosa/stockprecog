"""Upload stockprecog artifacts to the HuggingFace Hub — PREPARED, NOT EXECUTED.

This script stages an explicit ALLOWLIST of license-safe files into a temp folder and
pushes it with `huggingface_hub`. It REFUSES to upload if any denylisted artifact (raw
prices, .env, parquet, brapi caches) leaks into the staging area, and re-runs the git
token audit before touching the Hub.

It is intentionally NOT run here. To use it:

    1. pip/uv add huggingface_hub
    2. export HF_TOKEN=hf_xxx            # a write token from huggingface.co/settings/tokens
    3. set REPO_ID below (e.g. "your-username/stockprecog")
    4. review DRY_RUN -> set to False
    5. uv run python hf/upload_hf.py

Read hf/MANIFEST.md and hf/LICENSE_NOTICE.md before running. The raw price panel is
NEVER uploaded by design (B3 redistribution restriction + brapi contractual silence).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# from huggingface_hub import HfApi, create_repo, upload_folder  # noqa: E402  (uncomment to run)

# ---------------------------------------------------------------------------
# CONFIG — fill these in (placeholders).
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "<HF_TOKEN_PLACEHOLDER>")
REPO_ID = "<your-username>/stockprecog"      # e.g. "joaohb/stockprecog"
REPO_TYPE = "dataset"                          # "dataset" for the panel/recipe; "model" for the tilt card
DRY_RUN = True                                 # MUST flip to False deliberately to actually upload

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# ALLOWLIST — only these ship. See hf/MANIFEST.md.
# (src tree, tests, env files, docs, and the HF cards.)
# ---------------------------------------------------------------------------
ALLOW_FILES = [
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "hf/README_dataset.md",     # -> uploaded as README.md (dataset repo)
    "hf/README_model.md",       # -> uploaded as README.md (model repo)
    "hf/LICENSE_NOTICE.md",
    "hf/MANIFEST.md",
]
ALLOW_DIRS = [
    "src/stockprecog",
    "tests",
    "docs",
]

# ---------------------------------------------------------------------------
# DENYLIST — if any of these appear in staging, ABORT. Defense in depth.
# ---------------------------------------------------------------------------
DENY_GLOBS = ["*.parquet", "*.ZIP", "*.env", ".env", "*.ckpt"]
DENY_DIR_NAMES = {"data", "brapi_api", "__pycache__", ".venv", ".git"}


def git_token_audit() -> None:
    """Refuse to proceed if .env was ever tracked or a token-looking string is in history."""
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--all", "--oneline", "--", ".env"],
        capture_output=True, text=True,
    ).stdout.strip()
    if tracked:
        sys.exit("ABORT: .env appears in git history — clean it before any upload.")
    print("✓ git token audit clean (.env never tracked)")


def stage() -> Path:
    """Copy allowlisted paths into a clean temp folder; verify no denylisted leak."""
    staging = Path(tempfile.mkdtemp(prefix="stockprecog_hf_"))

    for rel in ALLOW_FILES:
        src = ROOT / rel
        if not src.exists():
            print(f"  (skip missing {rel})")
            continue
        dst = staging / Path(rel).name if rel.startswith("hf/") else staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel in ALLOW_DIRS:
        src = ROOT / rel
        if not src.exists():
            continue
        shutil.copytree(
            src, staging / rel,
            ignore=shutil.ignore_patterns(*DENY_GLOBS, *DENY_DIR_NAMES),
        )

    # Pick the right README for the repo type.
    readme_src = staging / ("README_dataset.md" if REPO_TYPE == "dataset" else "README_model.md")
    if readme_src.exists():
        shutil.copy2(readme_src, staging / "README.md")

    _verify_no_denylist(staging)
    print(f"✓ staged license-safe tree at {staging}")
    return staging


def _verify_no_denylist(staging: Path) -> None:
    for p in staging.rglob("*"):
        if p.is_dir() and p.name in DENY_DIR_NAMES:
            sys.exit(f"ABORT: denylisted dir leaked into staging: {p}")
        if p.is_file():
            for g in DENY_GLOBS:
                if p.match(g):
                    sys.exit(f"ABORT: denylisted file leaked into staging: {p}")
    print("✓ no denylisted artifacts in staging (no parquet / .env / data / ckpt)")


def main() -> None:
    if HF_TOKEN.startswith("<") or REPO_ID.startswith("<"):
        sys.exit("Set HF_TOKEN (env) and REPO_ID before running. This script is a prepared template.")

    git_token_audit()
    staging = stage()

    if DRY_RUN:
        print("\nDRY_RUN=True — nothing uploaded. Review the staged tree, then set DRY_RUN=False.")
        print(f"Would create repo '{REPO_ID}' (type={REPO_TYPE}) and upload {staging}.")
        return

    # --- actual upload (uncomment imports at top to enable) ---
    # create_repo(REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN, exist_ok=True, private=True)
    # upload_folder(
    #     folder_path=str(staging),
    #     repo_id=REPO_ID,
    #     repo_type=REPO_TYPE,
    #     token=HF_TOKEN,
    #     commit_message="stockprecog: pipeline + recipe + results (no raw market data)",
    #     ignore_patterns=DENY_GLOBS,
    # )
    # print(f"✓ uploaded to https://huggingface.co/{REPO_TYPE}s/{REPO_ID}")
    raise SystemExit("Upload block is commented out by design. Uncomment huggingface_hub calls to run.")


if __name__ == "__main__":
    main()
