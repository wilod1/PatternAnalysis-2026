"""
Smoke tests for the training loop on synthetic data (no ADNI data needed).
"""
import json
import math
import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import build_model, build_scheduler, fit, parse_args, set_seed

CPU = torch.device("cpu")
# The CNN keeps the real 256x256 input size; the others use 64x64 to stay fast.
SIZE = {"cnn": 256, "resnet": 64, "convnext": 64}


def make_batch(n, size, seed=0):
    """
    Synthetic two-class images: noise plus a class-dependent brightness offset,
    so a working model must be able to fit them quickly.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = 0.5 * torch.randn(n, 1, size, size, generator=g)
    x = x + (y.float() * 2 - 1).view(-1, 1, 1, 1)
    return x, y


def fake_loaders(size):
    """
    Train, val and test loaders with 16 images each (2 batches of 8).
    """
    def loader(seed):
        x, y = make_batch(16, size, seed)
        return DataLoader(TensorDataset(x, y), batch_size=8)
    return {"train": loader(0), "val": loader(1), "test": loader(2)}


@pytest.mark.parametrize("name", ["cnn", "resnet", "convnext"])
def test_fit_runs_and_writes_outputs(name, tmp_path):
    args = parse_args(["--model", name, "--epochs", "2", "--schedule", "cosine",
                       "--warmup-epochs", "1", "--label-smoothing", "0.1",
                       "--eval-test"])
    set_seed(0)
    summary = fit(build_model(name, args), fake_loaders(SIZE[name]), args, name,
                  CPU, out_dir=str(tmp_path))
    hist = summary["history"]
    assert len(hist["train_loss"]) == 2
    assert all(math.isfinite(v) for v in hist["train_loss"] + hist["val_loss"])
    assert hist["lr"][0] > hist["lr"][1]  # cosine: lr has started to decay
    assert "test_acc" in summary

    with open(tmp_path / "results" / f"{name}_history.json") as f:
        assert len(json.load(f)["history"]["epoch_time"]) == 2
    assert os.path.exists(tmp_path / "assets" / f"{name}_curves.png")

    ckpt = torch.load(tmp_path / "checkpoints" / f"{name}_best.pt",
                      map_location=CPU)
    build_model(ckpt["model"], args).load_state_dict(ckpt["state_dict"])


def test_cosine_schedule_warms_up_then_decays():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, "cosine", epochs=10, steps_per_epoch=10,
                            warmup_epochs=2)
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < lrs[10] <= 1.0                       # warmup rises
    assert all(a >= b for a, b in zip(lrs[19:], lrs[20:]))  # then decays
    assert lrs[-1] < 0.01
    assert build_scheduler(opt, "constant", 10, 10, 2) is None


@pytest.mark.parametrize("name", ["cnn", "resnet", "convnext"])
def test_model_fits_a_single_batch(name):
    """
    Sanity check: every model reaches >= 95% accuracy on one fixed batch.
    A failure here means a broken forward pass, loss or optimiser step.
    """
    args = parse_args(["--model", name, "--drop-path", "0.0"])
    set_seed(0)
    model = build_model(name, args)
    x, y = make_batch(8, SIZE[name])
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    for _ in range(100):
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    assert acc >= 0.95