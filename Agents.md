# Agents.md

## Project purpose
Build a reproducible workflow to record the repository owner's own voice, prepare a transcript-aligned dataset, and fine-tune Edge0-AI Audio8 TTS for clear English narration of short financial explainer videos.

## Current scope
- Record one utterance per WAV file using the laptop microphone.
- Preserve the exact prompt transcript with each recording.
- Perform basic audio checks and build deterministic train/evaluation raw JSONL manifests.
- Follow the upstream Audio8 preparation and SFT implementation rather than reimplementing its codec or trainer.
- Keep recordings, prepared codec arrays, checkpoints, and exports private and out of Git by default.

## Upstream contract
The upstream guide defines raw manifest rows with required `id`, `text`, and `audio` fields. `reference_audio` and `reference_text` are optional and must be supplied together. The upstream `audio8_tts_prepare.py` converts raw WAV audio to codec indices; `audio8_tts_sft.py` consumes the prepared manifest. Verify the upstream guide and script versions before changing this interface.

Primary upstream references:
- https://github.com/Edge0-AI/Audio8_TTS/blob/master/guides/training.md
- https://github.com/Edge0-AI/Audio8_TTS/blob/master/audio8_tts_prepare.py
- https://github.com/Edge0-AI/Audio8_TTS/blob/master/audio8_tts_sft.py

## Recording standards
- Default capture format: mono, 44,100 Hz, PCM-16 WAV.
- Record in a quiet room with a stable microphone distance and gain.
- Speak naturally; do not whisper, shout, or perform an exaggerated announcer voice.
- Read prompts verbatim. Redo any take with a changed or uncertain transcript.
- Avoid clipping, background music, reverberation, keyboard noise, and aggressive noise suppression.
- Keep utterances as separate clips; target roughly 2–10 seconds and avoid clips longer than 20 seconds.
- The recorder's RMS/peak checks are warnings, not a substitute for listening.

## Dataset rules
- Use a deterministic seed for train/eval splitting.
- Keep validation samples out of training.
- Inspect and listen to all clips before codec preparation.
- Add prompt diversity relevant to the target use: financial terms, numbers, percentages, dates, currency, questions, transitions, and sentence-final pauses.
- Do not invent sample-rate requirements. Check the selected checkpoint's codec config; the recorder defaults to the upstream 44.1 kHz codec rate for the 0.6B family.

## Engineering rules
- Python 3.10+.
- Keep recording dependencies separate from GPU training dependencies.
- Do not commit personal audio, transcripts that expose private information, prepared arrays, model weights, caches, or logs.
- Never claim training or export succeeded unless the corresponding command was actually run and verified.
- If upstream changes its manifest schema or trainer CLI, update this repository only after inspecting the new upstream code and guide.

## Planned next steps
1. Expand the recording prompt set and complete a first clean recording session.
2. Listen to the dataset and correct transcript mismatches.
3. Generate train/eval manifests.
4. Run upstream codec preparation on a CUDA-capable machine.
5. Run a small SFT experiment, evaluate held-out sentences, and compare generated speech with the original voice.
6. Store artifacts privately or with Git LFS if explicitly requested.
