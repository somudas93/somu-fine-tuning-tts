# Model artifacts

Run `python scripts/download_base_model.py` to download the official Audio8-TTS-Realistic-Air checkpoint into `models/base/audio8-realistic-air/`.

The snapshot is approximately 2.6 GB and includes the model weights, codec, tokenizer, processor, configuration, and custom model code. It is downloaded from Hugging Face; it is not copied into this Git repository.

Downloaded weights are excluded from Git by `.gitignore`. This avoids putting multi-gigabyte binary files into ordinary Git history and avoids unexpectedly publishing model files. If you explicitly want to version model binaries, use Git LFS or a private model registry and check storage/bandwidth costs first.

Upstream model: https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air
License shown by the upstream model card: Apache-2.0. Retain the upstream LICENSE and NOTICE files with any redistribution.


## Uploading the base checkpoint with Git LFS

The repository tracks large model binaries using `.gitattributes`. After running the downloader, the files under `models/base/audio8-realistic-air/` can be added and pushed with Git LFS. Keep personal recordings, prepared datasets, training checkpoints, and fine-tuned exports out of the repository.

From the repository root on your Mac:

```bash
git lfs install
git lfs track "*.safetensors"
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.bin"
python scripts/download_base_model.py
git add .gitattributes .gitignore models/base/audio8-realistic-air
git commit -m "Add Audio8 base model artifacts via Git LFS"
git push origin main
```

The `.gitattributes` rules are already committed. The `git lfs track` commands are safe to repeat; they confirm the local Git LFS patterns. Before pushing, verify that the GitHub repository's LFS storage and bandwidth quota can accommodate the approximately 2.6 GB snapshot.
