# Somu – Audio8 TTS fine-tuning

A personal-voice recording and fine-tuning workspace for [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS).

The goal is a clear, natural English narration voice for short financial explainer videos. This repository contains the recording and dataset-preparation utilities. It does **not** include your private voice recordings or trained model weights by default.

## 1. Set up the recording environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-recording.txt
```

If the microphone is not detected, list devices with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 2. Record your voice

```bash
python scripts/record_voice.py
```

The recorder reads prompts from `recording/prompts.tsv`. For each prompt, press Enter to start and Enter again to stop. You can keep, redo, or skip a take. Record in a quiet room, keep the microphone position consistent, and speak naturally. Do not imitate an announcer voice.

Output:
- `data/raw/*.wav`: mono 44.1 kHz PCM-16 audio
- `data/raw/recordings.jsonl`: exact transcript and file metadata

The prompts cover general narration, questions, pauses, currency, percentages, dates, and financial terms. Read each line exactly as written. If you stumble, redo it rather than keeping a mismatched transcript.

## 3. Validate and split the dataset

After recording at least five usable clips:

```bash
python scripts/prepare_manifests.py
```

This checks that files exist, are mono, have plausible durations, and have non-empty transcripts. It creates deterministic train/eval raw manifests under `data/manifests/`.

Aim for at least 30–60 minutes of clean speech for an initial experiment, then expand if results need improvement. This is a practical starting target, not a guarantee of quality. Keep validation sentences out of the training split.

## 4. Download the base model artifacts

Install the Hugging Face downloader (included in `requirements-recording.txt`) and run:

```bash
python scripts/download_base_model.py
```

This downloads the official `Edge0/Audio8-TTS-Realistic-Air` snapshot (about 2.6 GB) into `models/base/audio8-realistic-air/`. It includes the main weights, codec checkpoint, tokenizer, processor, configuration, and model code. The downloaded binaries stay in your local repository working directory but are excluded from Git by default. The upstream model card lists Apache-2.0; retain its LICENSE and NOTICE if redistributing. [Model card and file listing](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air).

## 5. Apple Silicon: recording, inference, and fine-tuning

Your M5 Pro is suitable for the recording workflow. Apple Silicon can also run some PyTorch workloads through MPS, but the official Audio8 SFT launcher currently relies on a CUDA/DeepSpeed-oriented distributed training stack. Do not assume that launcher will train on MPS. We should first check your Mac's unified memory and run a small model/codec loading test. If MPS training is unsupported or unstable, keep the Mac for recording and data preparation and use a CUDA GPU for SFT.

## 6. Audio8 codec preparation and SFT

Clone the upstream repository separately:

```bash
git clone https://github.com/Edge0-AI/Audio8_TTS.git
cd Audio8_TTS
pip install -r requirements-train.txt
```

Copy or point the raw manifests and WAV files into the Audio8 workspace while preserving relative paths. Then, from the Audio8_TTS repository root:

```bash
MODEL=Edge0/Audio8-TTS-Realistic-Air
python audio8_tts_prepare.py --model "$MODEL" \
  --input-jsonl /path/to/data/manifests/train.jsonl \
  --output-jsonl /path/to/prepared_data/train.jsonl

python audio8_tts_prepare.py --model "$MODEL" \
  --input-jsonl /path/to/data/manifests/eval.jsonl \
  --output-jsonl /path/to/prepared_data/eval.jsonl

MODEL="$MODEL" TRAIN_JSONL=/path/to/prepared_data/train.jsonl \
EVAL_JSONL=/path/to/prepared_data/eval.jsonl \
bash audio8_tts_sft.sh
```

Use the exact upstream SFT guide for GPU requirements, batch sizing, resume settings, and export options: [Audio8 training guide](https://github.com/Edge0-AI/Audio8_TTS/blob/master/guides/training.md).

The official 0.6B SFT pipeline is documented for CUDA-capable GPUs. Your M5 Pro may be able to load/run portions through PyTorch MPS, depending on unified memory and operator support, but the upstream DeepSpeed-based SFT launcher is not an Apple Silicon training recipe. Verify compatibility before planning local fine-tuning.

## Privacy and repository hygiene

- Voice recordings are personal biometric-like data. Keep `data/raw/`, prepared codec arrays, checkpoints, and exports out of Git.
- Do not upload voice samples to a public repository.
- Model checkpoints can be large; use private storage or Git LFS only if you explicitly decide to publish/store them.
- Confirm the model/checkpoint license and your intended use before distribution.

## Project status

The recording and raw-manifest utilities are the first implementation stage. Fine-tuning has not been run and no model artifact exists yet.
