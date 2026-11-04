#!/usr/bin/env python3
"""Record a personal-voice corpus for Audio8-TTS SFT.

Controls: Enter = start/stop current prompt, r = redo, s = skip, q = quit.
Audio is saved as mono 44.1 kHz PCM-16 WAV. Keep the transcript exact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "recording" / "prompts.tsv"
DEFAULT_OUT = ROOT / "data" / "raw"


def load_prompts(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"{path}:{line_no}: expected ID<TAB>sentence")
        rows.append((parts[0].strip(), parts[1].strip()))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def record_one(sample_rate: int, device: int | None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    print("Recording... speak naturally. Press Enter to stop.")
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=0,
        callback=lambda indata, frames, timing, status: chunks.append(indata.copy()),
    ):
        input()
    if not chunks:
        raise RuntimeError("No audio captured. Check microphone permissions/device.")
    return np.concatenate(chunks, axis=0).reshape(-1)


def assess(audio: np.ndarray, sample_rate: int) -> list[str]:
    problems = []
    duration = len(audio) / sample_rate
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if duration < 1.0:
        problems.append(f"very short recording ({duration:.1f}s)")
    if duration > 20.0:
        problems.append(f"long recording ({duration:.1f}s); consider splitting")
    if peak >= 0.999:
        problems.append("clipping detected")
    if rms < 0.003:
        problems.append("very low signal level / possible silence")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", type=int, default=None, help="Input device index; default system microphone")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "recordings.jsonl"
    existing = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row

    print(f"Prompts: {len(prompts)} | sample rate: {args.sample_rate} Hz")
    print("Use a quiet room, keep the laptop/mic position fixed, and speak conversationally.")
    print("For each prompt: Enter to record, Enter to stop; r redo; s skip; q quit.\n")

    for index, (sample_id, text) in enumerate(prompts, 1):
        target = args.output_dir / f"{sample_id}.wav"
        if target.exists() and sample_id in existing:
            print(f"[{index}/{len(prompts)}] {sample_id} already recorded; press Enter to keep, r to redo, s to skip, q to quit.")
        else:
            print(f"[{index}/{len(prompts)}] {sample_id}")
        print(f'TEXT: "{text}"')
        action = input("Action [Enter=record/keep, r=redo, s=skip, q=quit]: ").strip().lower()
        if action == "q":
            break
        if action == "s":
            continue
        if action == "" and target.exists() and sample_id in existing:
            continue

        while True:
            input("Press Enter when ready to start...")
            try:
                audio = record_one(args.sample_rate, args.device)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return 130
            issues = assess(audio, args.sample_rate)
            if issues:
                print("Quality warning: " + "; ".join(issues))
            preview = input("Keep this take? [y=keep, r=redo, s=skip]: ").strip().lower()
            if preview == "r":
                continue
            if preview != "y":
                break
            sf.write(target, audio, args.sample_rate, subtype="PCM_16")
            existing[sample_id] = {
                "id": sample_id,
                "text": text,
                "audio": target.name,
                "sample_rate": args.sample_rate,
                "duration_seconds": round(len(audio) / args.sample_rate, 3),
            }
            break

        manifest.write_text(
            "".join(json.dumps(existing[key], ensure_ascii=False) + "\n" for key in sorted(existing)),
            encoding="utf-8",
        )
        print(f"Saved progress: {len(existing)} accepted recordings\n")

    manifest.write_text(
        "".join(json.dumps(existing[key], ensure_ascii=False) + "\n" for key in sorted(existing)),
        encoding="utf-8",
    )
    print(f"Finished. Accepted recordings: {len(existing)}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
