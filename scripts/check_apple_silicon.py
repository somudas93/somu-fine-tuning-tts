#!/usr/bin/env python3
"""Check Apple Silicon MPS and DeepSpeed readiness for Audio8 SFT."""
from __future__ import annotations
import argparse
import platform
import sys

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-bf16', action='store_true', help='Fail if BF16 MPS forward/backward is unavailable')
    args = parser.parse_args()
    bf16_ok = False
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
    except Exception as exc:
        print(f"FAIL: PyTorch import failed: {exc}")
        return 1
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        print("FAIL: MPS unavailable. Install Apple-Silicon PyTorch and check macOS.")
        return 2
    device = torch.device("mps")
    try:
        x = torch.randn((512, 512), device=device, dtype=torch.float32, requires_grad=True)
        (x @ x).mean().backward()
        torch.mps.synchronize()
        print("PASS: FP32 MPS forward/backward")
        try:
            xb = torch.randn((128, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            (xb @ xb).float().mean().backward()
            torch.mps.synchronize()
            print("PASS: BF16 MPS forward/backward")
            bf16_ok = True
        except Exception as exc:
            print(f"WARN: BF16 test failed: {type(exc).__name__}: {exc}")
            print("Use BF16=false for the first SFT attempt.")
            if args.require_bf16:
                return 6
    except Exception as exc:
        print(f"FAIL: MPS forward/backward failed: {type(exc).__name__}: {exc}")
        return 3
    try:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        accelerator = get_accelerator()
        name = accelerator.device_name()
        print(f"DeepSpeed: {deepspeed.__version__}")
        print(f"DeepSpeed accelerator: {name}")
        if str(name).lower() not in {"mps", "mps:0"}:
            print("WARN: DeepSpeed did not select MPS.")
            return 4
        print("PASS: DeepSpeed detects MPS")
    except Exception as exc:
        print(f"WARN: DeepSpeed MPS check unavailable: {type(exc).__name__}: {exc}")
        print("Install DeepSpeed with DS_ACCELERATOR=mps and --no-build-isolation.")
        return 5
    print("Environment smoke tests passed; full Audio8 training still needs a real run.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
