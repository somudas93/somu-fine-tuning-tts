#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUDIO8_REPO="\${AUDIO8_REPO:-$ROOT/../Audio8_TTS}"
MODEL="\${MODEL:-$ROOT/models/base/audio8-realistic-air}"
if [[ ! -d "$AUDIO8_REPO" ]]; then
  echo "Audio8 source checkout not found: $AUDIO8_REPO" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "Base model not found: $MODEL" >&2
  echo "Run python scripts/download_base_model.py first." >&2
  exit 2
fi
cd "$AUDIO8_REPO"
python audio8_tts_prepare.py \
  --model "$MODEL" --device mps --dtype float32 \
  --input-jsonl "$ROOT/data/manifests/train.jsonl" \
  --output-jsonl "$ROOT/data/prepared/train.jsonl" --batch-size 1
python audio8_tts_prepare.py \
  --model "$MODEL" --device mps --dtype float32 \
  --input-jsonl "$ROOT/data/manifests/eval.jsonl" \
  --output-jsonl "$ROOT/data/prepared/eval.jsonl" --batch-size 1
