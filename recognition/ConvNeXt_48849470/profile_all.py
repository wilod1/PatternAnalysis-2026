"""
Resource profile of the models compared in the report: parameter count, peak GPU
memory for one training step, peak GPU memory for inference, and inference latency.

Every model is profiled with the settings of the headline experiments: 1-channel
224 x 224 input, batch 64 for training and for batched inference, mixed precision.
Needs a GPU. Writes results/final/profile.json and prints a table.

Usage:  python profile_all.py
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from modules import CONVNEXT_VARIANTS, ConvNeXt, SimpleCNN, SimpleResNet

OUT_DIR = Path("results/final")
SIZE = 224
BATCH = 64
GIB = 2 ** 30


def build_models():
    def convnext(variant):
        return ConvNeXt(in_channels=1, num_classes=2, drop_path_rate=0.1, dropout=0.0,
                        **CONVNEXT_VARIANTS[variant])

    return {
        "convnext_tiny": convnext("tiny"),
        "convnext_mini": convnext("mini"),
        "resnet": SimpleResNet(in_channels=1, num_classes=2),
        "cnn": SimpleCNN(in_channels=1, num_classes=2),
    }


def training_peak(model, device):
    """Peak allocated memory (GiB) over one AMP training step at batch 64."""
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(BATCH, 1, SIZE, SIZE, device=device)
    labels = torch.randint(0, 2, (BATCH,), device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GIB


@torch.no_grad()
def inference_profile(model, device):
    """Peak memory (GiB) and latency (ms) for inference under autocast."""
    model = model.to(device).eval()
    out = {}
    for batch in (1, BATCH):
        images = torch.randn(batch, 1, SIZE, SIZE, device=device)
        for _ in range(20):  # warm-up
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(images)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(100 if batch == 1 else 50):
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(images)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        median = times[len(times) // 2]
        out[f"batch{batch}_median_ms"] = median
        out[f"batch{batch}_ms_per_slice"] = median / batch
        out[f"batch{batch}_peak_gib"] = torch.cuda.max_memory_allocated() / GIB
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("profile_all.py needs a GPU; run it inside a Slurm GPU job")
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    results = {}
    for name, model in build_models().items():
        params = sum(p.numel() for p in model.parameters())
        train_gib = training_peak(model, device)
        infer = inference_profile(model, device)
        results[name] = {"parameters": params, "train_peak_gib": train_gib, **infer}
        model.cpu()
        torch.cuda.empty_cache()

    print(f"\n{'model':15s} {'parameters':>12s} {'train peak GiB':>15s} "
          f"{'infer peak GiB (b64)':>21s} {'ms/slice (b1)':>14s} {'ms/slice (b64)':>15s}")
    for name, r in results.items():
        print(f"{name:15s} {r['parameters']:12,d} {r['train_peak_gib']:15.2f} "
              f"{r['batch64_peak_gib']:21.2f} {r['batch1_ms_per_slice']:14.2f} "
              f"{r['batch64_ms_per_slice']:15.3f}")

    with open(OUT_DIR / "profile.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/final/profile.json")


if __name__ == "__main__":
    main()