#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUDIO8_REPO="${AUDIO8_REPO:-$ROOT/../Audio8_TTS}"
MODEL="${MODEL:-$ROOT/models/base/audio8-realistic-air}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT/data/prepared/train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-$ROOT/data/prepared/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/audio8-m5pro-sft}"
EXPORT_DIR="${EXPORT_DIR:-$ROOT/models/finetuned/audio8-m5pro-voice}"

if [[ ! -d "$AUDIO8_REPO" ]]; then
  echo "Audio8 source checkout not found: $AUDIO8_REPO" >&2
  echo "Clone https://github.com/Edge0-AI/Audio8_TTS or set AUDIO8_REPO." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Prepared training manifest not found: $TRAIN_JSONL" >&2
  echo "Run codec preparation first; see README.md." >&2
  exit 2
fi
if [[ ! -f "$AUDIO8_REPO/configs/deepspeed_zero2.json" ]]; then
  echo "Missing upstream DeepSpeed ZeRO-2 config." >&2
  exit 2
fi

export DS_ACCELERATOR=mps
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
if [[ ! -f "$EVAL_JSONL" ]]; then EVAL_JSONL=""; fi
export MODEL TRAIN_JSONL EVAL_JSONL OUTPUT_DIR EXPORT_DIR
export NPROC_PER_NODE=1
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export BF16="${BF16:-true}"
export GRADIENT_CHECKPOINTING=false
export DATALOADER_NUM_WORKERS=0
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

if [[ "$BF16" == "true" ]]; then
  python "$ROOT/scripts/check_apple_silicon.py" --require-bf16
else
  python "$ROOT/scripts/check_apple_silicon.py"
fi
cd "$AUDIO8_REPO"
bash ./audio8_tts_sft.sh
