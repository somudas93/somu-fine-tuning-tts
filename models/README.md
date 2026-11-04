# Model artifacts

Run `python scripts/download_base_model.py` to download the official Audio8-TTS-Realistic-Air checkpoint into `models/base/audio8-realistic-air/`.

The snapshot is approximately 2.6 GB and includes the model weights, codec, tokenizer, processor, configuration, and custom model code. It is downloaded from Hugging Face; it is not copied into this Git repository.

Downloaded weights are excluded from Git by `.gitignore`. This avoids putting multi-gigabyte binary files into ordinary Git history and avoids unexpectedly publishing model files. If you explicitly want to version model binaries, use Git LFS or a private model registry and check storage/bandwidth costs first.

Upstream model: https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air
License shown by the upstream model card: Apache-2.0. Retain the upstream LICENSE and NOTICE files with any redistribution.
