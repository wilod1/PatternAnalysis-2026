import numpy as np
import torch
from PIL import Image

from predict import load_model, predict_images
from train import build_model, parse_args


def test_checkpoint_roundtrip_and_prediction(tmp_path):
    args = parse_args(["--model", "cnn"])
    model = build_model("cnn", args)
    ckpt = tmp_path / "cnn_best.pt"
    torch.save({"model": "cnn", "state_dict": model.state_dict(),
                "args": vars(args), "epoch": 1}, ckpt)
    img = tmp_path / "x.png"
    pixels = (np.random.RandomState(0).rand(200, 180) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(img)

    cpu = torch.device("cpu")
    loaded, saved = load_model(str(ckpt), cpu)
    assert saved["img_size"] == args.img_size
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)

    probs = predict_images(loaded, [str(img), str(img)], 0.1, 0.2,
                           args.img_size, cpu)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(probs[0], probs[1])