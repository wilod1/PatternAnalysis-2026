import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import (ModelEMA, args_from_checkpoint, build_model, fit,
                   mixup_batch, parse_args, train_one_epoch)

CPU = torch.device("cpu")


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def tiny_loader(n=16, size=64, batch=8):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, size, size, generator=g)
    y = torch.arange(n) % 2
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def test_mixup_alpha_zero_is_identity():
    x, y = torch.randn(4, 1, 8, 8), torch.tensor([0, 1, 0, 1])
    xm, ya, yb, lam = mixup_batch(x, y, 0.0)
    assert torch.equal(xm, x) and torch.equal(ya, y) and torch.equal(yb, y)
    assert lam == 1.0


def test_mixup_blends_inputs_with_the_partner_labels():
    # row i is a constant image of value i and has label i, so the blended
    # value of row i must be lam * label_a + (1 - lam) * label_b
    n = 6
    x = torch.arange(n).float().view(n, 1, 1, 1).expand(n, 1, 4, 4).clone()
    y = torch.arange(n)
    torch.manual_seed(0)
    xm, ya, yb, lam = mixup_batch(x, y, 0.4)
    assert 0.0 <= lam <= 1.0
    expected = lam * ya.float() + (1 - lam) * yb.float()
    assert torch.allclose(xm[:, 0, 0, 0], expected, atol=1e-5)
    assert sorted(yb.tolist()) == list(range(n))  # a permutation


def test_ema_converges_and_is_independent_of_the_model():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.9)
    before = ema.module.weight.clone()
    with torch.no_grad():
        model.weight.add_(5.0)
    assert torch.equal(ema.module.weight, before)  # not updated yet
    assert not any(p.requires_grad for p in ema.module.parameters())
    for _ in range(200):
        ema.update(model)
    assert torch.allclose(ema.module.weight, model.weight, atol=1e-3)


def test_ema_copies_buffers():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    model.train()
    model(torch.randn(16, 2) + 3.0)  # moves the running statistics
    ema = ModelEMA(model, decay=0.9)
    ema.update(model)
    assert torch.equal(ema.module[1].running_mean, model[1].running_mean)


def test_build_model_depths_and_dims():
    default = build_model("convnext", parse_args(["--model", "convnext"]))
    assert n_params(default) == 1899874
    args = parse_args(["--model", "convnext", "--depths", "1", "1", "3", "1",
                       "--dims", "24", "48", "96", "192"])
    small = build_model("convnext", args)
    assert 0 < n_params(small) < 1_000_000
    assert small(torch.randn(1, 1, 64, 64)).shape == (1, 2)


def test_args_from_checkpoint_restores_architecture():
    ckpt = {"model": "convnext", "args": {
        "depths": [1, 1, 3, 1], "dims": [24, 48, 96, 192],
        "drop_path": 0.0, "not_an_option": 5}}
    small = build_model("convnext", args_from_checkpoint(ckpt))
    assert n_params(small) < 1_000_000
    legacy = {"model": "convnext", "args": {"lr": 0.1}}  # older checkpoint
    old = build_model("convnext", args_from_checkpoint(legacy))
    assert n_params(old) == 1899874


def test_train_one_epoch_with_mixup_and_ema():
    args = parse_args(["--model", "cnn"])
    model = build_model("cnn", args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, 0.9)
    loss, acc = train_one_epoch(model, tiny_loader(), opt, nn.CrossEntropyLoss(),
                                CPU, mixup_alpha=0.4, ema=ema)
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0
    assert ema.updates == 2


def test_fit_with_mixup_and_ema_writes_a_loadable_checkpoint(tmp_path):
    args = parse_args(["--model", "cnn", "--epochs", "1", "--mixup", "0.4",
                       "--ema-decay", "0.9"])
    model = build_model("cnn", args)
    loaders = {"train": tiny_loader(), "val": tiny_loader()}
    summary = fit(model, loaders, args, "t", CPU, out_dir=str(tmp_path))
    assert math.isfinite(summary["best_val_acc"])
    ckpt = torch.load(tmp_path / "checkpoints" / "t_best.pt",
                      map_location="cpu", weights_only=False)
    fresh = build_model("cnn", args_from_checkpoint(ckpt))
    fresh.load_state_dict(ckpt["state_dict"])