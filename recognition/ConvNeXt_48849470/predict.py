"""
Classify ADNI slices with a trained model.

Prints, for each slice, the predicted class (NC or AD) and the probability the
model gives to each class. When the slice comes from the validation set (see
--demo) the true label is shown too and each prediction is marked correct or
WRONG.

    python predict.py --run convnext_b --image path/to/slice.jpeg
    python predict.py --run convnext_b --demo 6

Arguments
    --run NAME     name of a trained run; loads checkpoints/NAME_best.pt
    --image PATH   one or more slice files (JPEG, any size the dataset uses)
    --demo N       instead of files, classify N random validation slices

The evaluation preprocessing (padding and normalisation) is rebuilt from the
settings stored in the checkpoint, so a slice is prepared exactly as it was
during training.
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

# checkpoints/ is looked up next to this file, whatever the working directory is
HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    """
    Rebuild a trained model from a checkpoint; returns (model, saved args dict).
    """
    # weights_only=False: the checkpoint stores the training settings as well
    # as the weight tensors, and those settings are plain Python objects
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # settings are stored either as a dict or as an argparse namespace; turn
    # both into a dict so the caller can read them the same way
    saved = ckpt["args"] if isinstance(ckpt["args"], dict) else vars(ckpt["args"])
    name = ckpt["model"]  # which architecture the weights belong to
    # build the architecture from the saved settings, then fill in the weights
    model = build_model(name, args_from_checkpoint(ckpt))
    model.load_state_dict(ckpt["state_dict"])
    # eval() switches off dropout and stochastic depth for prediction
    return model.to(device).eval(), saved


@torch.no_grad()  # inference only: no gradients are needed
def predict_images(model, paths, mean, std, img_size, device):
    """
    Class probabilities for image files, using the evaluation preprocessing.
    """
    # evaluation pipeline: padding/resizing and normalisation, no augmentation
    tf = build_transforms(False, mean, std, img_size)
    # for each file: open as one-channel grayscale ("L"), convert to a float
    # tensor in [0, 1], apply the pipeline, then stack into a single batch
    batch = torch.stack([tf(TF.to_tensor(Image.open(p).convert("L")))
                         for p in paths]).to(device)
    # raw logits -> probabilities that sum to 1 over (NC, AD)
    return torch.softmax(model(batch), dim=1).cpu()


def main():
    # command-line interface: a run name plus either image files or a demo count
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
    # the seed and image size the model was trained with
    seed, img_size = saved["seed"], saved["img_size"]
    # Rebuild the training data split and the normalisation statistics with the
    # same seed as training. The statistics must come from the training images
    # only, because that is what the model was trained against.
    splits = split_subjects(collect_samples(), seed=seed)
    mean, std = compute_norm_stats(splits["train"], seed=seed)

    if a.demo:
        # a reproducible random sample of validation slices, with known labels
        records = random.Random(seed).sample(splits["val"], a.demo)
    else:
        # files given by the user have no known label
        records = [{"path": path, "label": None} for path in a.image]
    probs = predict_images(model, [r["path"] for r in records],
                           mean, std, img_size, device)
    # one output line per slice: file name, predicted class and both probabilities
    for r, pr in zip(records, probs):
        pred = int(pr.argmax())  # index of the more probable class (0 = NC, 1 = AD)
        line = (f"{os.path.basename(r['path'])}: predicted "
                f"{str(CLASS_NAMES[pred]).upper()} "
                f"({str(CLASS_NAMES[0]).upper()} {pr[0]:.3f}, "
                f"{str(CLASS_NAMES[1]).upper()} {pr[1]:.3f})")
        if r["label"] is not None:
            # a true label is known (demo mode): show it and whether the prediction matched
            ok = "correct" if pred == r["label"] else "WRONG"
            line += f"  true {str(CLASS_NAMES[r['label']]).upper()} [{ok}]"
        print(line)


if __name__ == "__main__":
    main()