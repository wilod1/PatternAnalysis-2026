"""
Cost of each architecture: parameters, multiply-accumulates (MACs), forward
latency, throughput and peak training memory. MACs count convolution and
linear layers only; norms, activations and additions are ignored.
"""
import argparse
import json
import os
import time

import torch
from torch import nn

from train import build_model, parse_args

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("cnn", "resnet", "convnext")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model, input_shape):
    """
    MACs for one forward pass of an input of input_shape (batch of 1).
    """
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += (out.numel() * (m.in_channels // m.groups)
                     * m.kernel_size[0] * m.kernel_size[1])

    def linear_hook(m, inp, out):
        total[0] += out.numel() * m.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))
    for h in handles:
        h.remove()
    model.train(was_training)
    return total[0]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_forward(model, batch, size, device, warmup=5, iters=20):
    """
    Mean seconds per forward pass at the given batch size.
    """
    model.eval()
    x = torch.randn(batch, 1, size, size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync(device)
    return (time.perf_counter() - start) / iters


def train_step_memory(model, batch, size, device):
    """
    Peak GPU memory (MiB) for one forward and backward pass, or None on CPU.
    """
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    x = torch.randn(batch, 1, size, size, device=device)
    y = torch.zeros(batch, dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def profile_model(name, device, img_size, batch=64, warmup=5, iters=20):
    args = parse_args(["--model", name])
    model = build_model(name, args).to(device)
    macs = count_macs(model, (1, 1, img_size, img_size))
    latency_1 = time_forward(model, 1, img_size, device, warmup, iters)
    latency_b = time_forward(model, batch, img_size, device, warmup, iters)
    return {
        "model": name,
        "params": count_params(model),
        "macs_millions": macs / 1e6,
        "latency_ms_batch1": latency_1 * 1000,
        "throughput_img_per_s": batch / latency_b,
        "train_memory_mib": train_step_memory(model, batch, img_size, device),
        "batch": batch,
        "img_size": img_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--img-size", type=int, default=parse_args([]).img_size)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (torch.cuda.get_device_name(device)
                   if device.type == "cuda" else "cpu")
    rows = []
    for name in a.models:
        r = profile_model(name, device, a.img_size, a.batch)
        rows.append(r)
        mem = ("n/a" if r["train_memory_mib"] is None
               else f"{r['train_memory_mib']:.0f} MiB")
        print(f"{name:9s} {r['params']:>9,d} params  "
              f"{r['macs_millions']:7.1f} M MACs  "
              f"{r['latency_ms_batch1']:6.2f} ms/img (batch 1)  "
              f"{r['throughput_img_per_s']:8.0f} img/s (batch {a.batch})  "
              f"train mem {mem}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "profile.json"), "w") as f:
        json.dump({"device": device_name, "models": rows}, f, indent=2)
    print(f"saved profile.json (device: {device_name})")


if __name__ == "__main__":
    main()