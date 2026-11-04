#!/usr/bin/env python3
"""Create Audio8 raw train/eval JSONL manifests from recorder output."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/recordings.jsonl"))
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    args = parser.parse_args()

    if not 0.05 <= args.eval_ratio <= 0.4:
        raise ValueError("--eval-ratio must be between 0.05 and 0.4")

    manifest = args.manifest.resolve()
    rows = []
    seen = set()
    for line_no, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"duplicate id at line {line_no}: {sample_id}")
        seen.add(sample_id)
        audio_path = (manifest.parent / row["audio"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        info = sf.info(audio_path)
        duration = float(info.duration)
        if info.channels != 1:
            raise ValueError(f"{audio_path}: expected mono audio, found {info.channels} channels")
        if not args.min_seconds <= duration <= args.max_seconds:
            raise ValueError(f"{audio_path}: duration {duration:.2f}s outside allowed range")
        if not str(row.get("text", "")).strip():
            raise ValueError(f"{audio_path}: empty transcript")
        rows.append({
            "id": sample_id,
            "text": row["text"].strip(),
            "audio": __import__("os").path.relpath(audio_path, output_dir).replace("\\", "/"),
        })

    if len(rows) < 5:
        raise ValueError(f"Need at least 5 accepted recordings before creating train/eval split; found {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    eval_count = max(1, round(len(rows) * args.eval_ratio))
    eval_rows, train_rows = rows[:eval_count], rows[eval_count:]

    output_dir = manifest.parent.parent / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.jsonl", train_rows), ("eval.jsonl", eval_rows)):
        path = output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(subset)} samples")

    print("These are Audio8 raw manifests. Next run audio8_tts_prepare.py from the Audio8_TTS checkout.")
    print("Do not run codec preparation until you have enough clean, transcript-matched recordings.")


if __name__ == "__main__":
    main()
