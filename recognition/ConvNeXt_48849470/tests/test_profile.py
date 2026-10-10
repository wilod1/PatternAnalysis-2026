import torch
from torch import nn

from profile_models import count_macs, count_params, profile_model


def test_count_macs_hand_example():
    # conv: 2 output channels x 4x4 positions x (1 in channel x 3x3) = 288
    conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
    assert count_macs(conv, (1, 1, 4, 4)) == 288
    # linear: 3 outputs x 4 inputs = 12
    assert count_macs(nn.Linear(4, 3), (1, 4)) == 12


def test_count_macs_depthwise_uses_groups():
    # depthwise: 4 channels x 4x4 positions x (1 in channel per group x 3x3) = 576
    dw = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4)
    assert count_macs(dw, (1, 4, 4, 4)) == 576


def test_count_macs_restores_training_mode():
    model = nn.Sequential(nn.Conv2d(1, 1, 3), nn.BatchNorm2d(1))
    model.train()
    count_macs(model, (1, 1, 8, 8))
    assert model.training


def test_param_counts_match_known_models():
    from train import build_model, parse_args
    for name, expected in (("resnet", 700146), ("convnext", 1899874)):
        model = build_model(name, parse_args(["--model", name]))
        assert count_params(model) == expected


def test_profile_model_runs_on_cpu():
    r = profile_model("cnn", torch.device("cpu"), img_size=32, batch=2,
                      warmup=1, iters=1)
    assert r["params"] > 0 and r["macs_millions"] > 0
    assert r["latency_ms_batch1"] > 0 and r["throughput_img_per_s"] > 0
    assert r["train_memory_mib"] is None