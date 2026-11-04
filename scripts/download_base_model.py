#!/usr/bin/env python3
"""Download the official Audio8 Realistic Air checkpoint into this workspace."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Edge0/Audio8-TTS-Realistic-Air"
DEFAULT_DESTINATION = ROOT / "models" / "base" / "audio8-realistic-air"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="Hugging Face branch, tag, or commit")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated/private repos")
    args = parser.parse_args()

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id}@{args.revision}")
    print(f"Destination: {destination}")
    print("The complete checkpoint is approximately 2.6 GB; ensure sufficient disk space.")

    downloaded = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(destination),
        token=args.token,
    )
    print(f"Model snapshot is ready at: {downloaded}")
    print("These files are local model artifacts; they are intentionally excluded from Git.")


if __name__ == "__main__":
    main()
