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

Your machine has an Apple M5 Pro with 48 GB unified memory. PyTorch supports GPU execution through Apple's MPS backend. Recent DeepSpeed documentation also describes single-device MPS training, including ZeRO stages 0–3, but its published hardware verification currently names an M5 Max rather than an M5 Pro. So this workflow is a **carefully configured local training attempt**, not a guarantee that every Audio8 operator and dependency will work on your exact Mac. Run the included smoke test before starting a training job. [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) · [DeepSpeed MPS setup and limitations](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/).

## 6. Audio8 codec preparation and SFT

Create a separate upstream checkout beside this repository:

```bash
cd ..
git clone https://github.com/Edge0-AI/Audio8_TTS.git
cd somu-fine-tuning-tts
```

Create an Apple-Silicon environment (Python 3.10+; use a fresh environment for this stack):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mps.txt
```

Install DeepSpeed using its documented MPS build path (not the ordinary training requirements file):

```bash
DS_ACCELERATOR=mps pip install --no-build-isolation deepspeed
```

Check the actual machine before processing data:

```bash
python scripts/check_apple_silicon.py
```

If the MPS and DeepSpeed checks pass, download the base model and record/prepare your dataset:

```bash
python scripts/download_base_model.py
python scripts/record_voice.py
python scripts/prepare_manifests.py
bash scripts/prepare_audio8_mps.sh
```

The codec-preparation wrapper explicitly selects `--device mps --dtype float32` and uses batch size 1. The upstream preparation script's automatic device selection defaults to CUDA or CPU, so do not omit the explicit MPS selection on this Mac.

Start a conservative single-device SFT run:

```bash
bash scripts/run_mps_sft.sh
```

Initial settings for 48 GB unified memory:
- one MPS device, microbatch size 1, gradient accumulation 8
- BF16 enabled only after the included smoke test passes
- no gradient checkpointing for the 0.6B Realistic Air model (upstream marks it unsupported)
- one data-loader worker; MPS fallback enabled for unsupported operators
- checkpoint and exported model saved under this repository's ignored `checkpoints/` and `models/finetuned/` directories

If BF16 smoke testing fails, retry with `BF16=false bash scripts/run_mps_sft.sh`. If DeepSpeed does not detect MPS, the model fails to load, or the first training step hits an unsupported operator/OOM, stop rather than repeatedly retrying; capture the full error and use a CUDA GPU for SFT if needed. The smoke test verifies framework operations and accelerator detection, not end-to-end model training.

For codec preparation or inference only, MPS may work even if DeepSpeed training does not. Use the wrappers and keep batch size small. See the upstream [Audio8 SFT guide](https://github.com/Edge0-AI/Audio8_TTS/blob/master/guides/training.md) and [DeepSpeed Apple Silicon instructions](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/).

## Privacy and repository hygiene

- Voice recordings are personal biometric-like data. Keep `data/raw/`, prepared codec arrays, checkpoints, and exports out of Git.
- Do not upload voice samples to a public repository.
- Model checkpoints can be large; use private storage or Git LFS only if you explicitly decide to publish/store them.
- Confirm the model/checkpoint license and your intended use before distribution.

## Project status

The recording and raw-manifest utilities are the first implementation stage. Fine-tuning has not been run and no model artifact exists yet.
