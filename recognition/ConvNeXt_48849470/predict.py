"""
Classify ADNI slices with a trained model.

    python predict.py --run convnext_b --image path/to/slice.jpeg
    python predict.py --run convnext_b --demo 6
"""
import argparse
import os
import random

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from dataset import (build_transforms, collect_samples, compute_norm_stats,
                     split_subjects)
from metrics import CLASS_NAMES
from train import args_from_checkpoint, build_model

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    """
    Rebuild a trained model from a checkpoint; returns (model, saved args dict).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt["args"] if isinstance(ckpt["args"], dict) else vars(ckpt["args"])
    name = ckpt["model"]
    model = build_model(name, args_from_checkpoint(ckpt))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), saved


@torch.no_grad()
def predict_images(model, paths, mean, std, img_size, device):
    """
    Class probabilities for image files, using the evaluation preprocessing.
    """
    tf = build_transforms(False, mean, std, img_size)
    batch = torch.stack([tf(TF.to_tensor(Image.open(p).convert("L")))
                         for p in paths]).to(device)
    return torch.softmax(model(batch), dim=1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--image", nargs="*", default=[])
    p.add_argument("--demo", type=int, default=0,
                   help="classify this many random validation slices")
    a = p.parse_args()
    if not a.image and not a.demo:
        p.error("give --image paths or --demo N")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved = load_model(
        os.path.join(HERE, "checkpoints", f"{a.run}_best.pt"), device)
    seed, img_size = saved["seed"], saved["img_size"]
    splits = split_subjects(collect_samples(), seed=seed)
    mean, std = compute_norm_stats(splits["train"], seed=seed)

    if a.demo:
        records = random.Random(seed).sample(splits["val"], a.demo)
    else:
        records = [{"path": path, "label": None} for path in a.image]
    probs = predict_images(model, [r["path"] for r in records],
                           mean, std, img_size, device)
    for r, pr in zip(records, probs):
        pred = int(pr.argmax())
        line = (f"{os.path.basename(r['path'])}: predicted "
                f"{str(CLASS_NAMES[pred]).upper()} "
                f"({str(CLASS_NAMES[0]).upper()} {pr[0]:.3f}, "
                f"{str(CLASS_NAMES[1]).upper()} {pr[1]:.3f})")
        if r["label"] is not None:
            ok = "correct" if pred == r["label"] else "WRONG"
            line += f"  true {str(CLASS_NAMES[r['label']]).upper()} [{ok}]"
        print(line)


if __name__ == "__main__":
    main()