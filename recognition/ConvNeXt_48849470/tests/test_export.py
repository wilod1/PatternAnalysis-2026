"""
Tests for prediction export, on synthetic data (no ADNI needed).
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from train import build_model, fit, parse_args, set_seed
from utils import RECORD_FIELDS, export_predictions, predict_split

CPU = torch.device("cpu")


class FakeSliceDataset(Dataset):
    """
    Random two-class images with a records list shaped like the real dataset's.
    """

    def __init__(self, n, size=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.y = torch.arange(n) % 2
        self.x = 0.5 * torch.randn(n, 1, size, size, generator=g)
        self.x = self.x + (self.y.float() * 2 - 1).view(-1, 1, 1, 1)
        self.records = [
            {"label": int(self.y[i]), "subject_id": f"subj{i // 4}",
             "scan_id": f"scan{i // 2}", "slice_index": i, "window": i % 3,
             "path": f"/fake/{i}.jpeg"} for i in range(n)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def test_predict_split_keeps_order_and_sums_to_one():
    args = parse_args(["--model", "resnet"])
    set_seed(0)
    model = build_model("resnet", args).eval()
    data = FakeSliceDataset(10)
    probs = predict_split(model, DataLoader(data, batch_size=3), CPU)  # uneven last batch
    with torch.no_grad():
        expected = torch.softmax(model(data.x), dim=1).numpy()
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)
    assert np.allclose(probs, expected, atol=1e-5)


def test_predict_split_rejects_shuffled_loader():
    model = build_model("resnet", parse_args(["--model", "resnet"]))
    loader = DataLoader(FakeSliceDataset(8), batch_size=4, shuffle=True)
    with pytest.raises(ValueError):
        predict_split(model, loader, CPU)


def test_export_roundtrip(tmp_path):
    args = parse_args(["--model", "resnet", "--epochs", "1"])
    set_seed(0)
    loaders = {s: DataLoader(FakeSliceDataset(16, seed=i), batch_size=8)
               for i, s in enumerate(("train", "val", "test"))}
    fit(build_model("resnet", args), loaders, args, "resnet", CPU,
        out_dir=str(tmp_path))

    out_path = tmp_path / "results" / "preds_resnet.npz"
    export_predictions(str(tmp_path / "checkpoints" / "resnet_best.pt"),
                       loaders, str(out_path), CPU)
    saved = np.load(out_path)  # allow_pickle stays off: arrays must be plain
    for split in ("val", "test"):
        probs = saved[f"{split}_probs"]
        assert probs.shape == (16, 2)
        assert np.allclose(probs.sum(1), 1.0, atol=1e-5)
        assert (saved[f"{split}_preds"] == probs.argmax(1)).all()
        for field in RECORD_FIELDS:
            assert len(saved[f"{split}_{field}"]) == 16
        expected = [r["label"] for r in loaders[split].dataset.records]
        assert saved[f"{split}_label"].tolist() == expected