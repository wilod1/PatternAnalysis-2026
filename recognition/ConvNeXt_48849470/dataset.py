"""
Data utilities for ADNI AD-vs-NC classification (2D brain MRI slices).

The privided train/test folders share patients, so all slices are pooled
here, and re-split by subject (please see split_subjects).
"""

import json
import os
import re
import random
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

ADNI_ROOT = "/home/groups/comp3710/ADNI"
JSON_NAME = "meta_data_with_label.json"
CLASS_TO_INDEX = {"NC": 0, "AD": 1} # model labels
JSON_LABEL_TO_CLASS = {0: "NC", 2: "AD"} # label 1 is not in the folders

# Subject IDs look like 068_S_0473 inside the JSON "raw" path
_SUBJECT_RE = re.compile(r"ADNI_(\d+_S_\d+)")


def load_scan_table(root=ADNI_ROOT):
    """
    Map scan ID -> {'subject_id': str, 'label': int} from the metadata JSON.
    """
    with open(os.path.join(root, JSON_NAME)) as f:
        meta = json.load(f)

    table = {}
    for scan_id, entry in meta.items():
        match = _SUBJECT_RE.search(entry["raw"])
        if match is None:
            raise ValueError(f"No subject ID found for scan {scan_id}")
        table[scan_id] = {"subject_id": match.group(1), "label": entry["label"]}

    return table


def collect_samples(root=ADNI_ROOT):
    """
    Pool the provided train/test folders into one record per slice.

    Each record holds: path, label (0=NC, 1=AD), scan_id, subject_id,
    slice_index and window (first, last slice index of the scan).
    """
    scans = load_scan_table(root)
    records = []
    for split in ("train", "test"):
        for cls, label in CLASS_TO_INDEX.items():
            folder = os.path.join(root, "AD_NC", split, cls)
            for name in sorted(os.listdir(folder)):
                scan_id, slice_idx = os.path.splitext(name)[0].split("_")
                info = scans[scan_id]

                # Sanity check -- folder class must agree with JSON label
                if JSON_LABEL_TO_CLASS[info["label"]] != cls:
                    raise ValueError(f"Label mismatch for scan {scan_id}")

                records.append({
                    "path": os.path.join(folder, name),
                    "label": label,
                    "scan_id": scan_id,
                    "subject_id": info["subject_id"],
                    "slice_index": int(slice_idx)
                })

    # Window = (first, last) sslice index of each scan
    spans = {}
    for r in records:
        lo, hi = spans.get(r["scan_id"], (10**9, -1))
        spans[r["scan_id"]] = (min(lo, r["slice_index"]), max(hi, r["slice_index"]))
    for r in records:
        r["window"] = spans[r["scan_id"]]
    return records

INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}
SEED = 42 # for reproducibility

def split_subjects(samples, seed=SEED, ratios=(0.70, 0.15, 0.15)):
    """
    Split slice records into train/val/test, stratified by class, by SUBJECT.

    Every slice of a subject (across all of their scans) goes to the same
    split, which prevents patient-level leakage. Each subject has a single
    class (checked below), so class is used as the stratum.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    # subject -> class label, verifying that a subject never has 2 classes
    subject_label = {}
    for r in samples:
        prev = subject_label.setdefault(r["subject_id"], r["label"])
        if prev != r["label"]:
            raise ValueError(f"Subject {r['subject_id']} has more than one class")

    rng = random.Random(seed)
    assignment = {}
    for label in sorted(set(subject_label.values())):
        # Sort before shuffling so result depends only on seed
        subjects = sorted(s for s, l in subject_label.items() if l == label)
        rng.shuffle(subjects)
        n_train = round(ratios[0] * len(subjects))
        n_val = round(ratios[1] * len(subjects))
        for i, s in enumerate(subjects):
            if i < n_train:
                assignment[s] = "train"
            elif i< n_train + n_val:
                assignment[s] = "val"
            else:
                assignment[s] = "test"

    splits = {"train": [], "val": [], "test": []}
    for r in samples:
        splits[assignment[r["subject_id"]]].append(r)
    return splits


def describe_split(splits):
    """
    Print subjects, scans, slices and window-by-class scan counts per split.
    """
    for name, recs in splits.items():
        subjects = {r["subject_id"]: r["label"] for r in recs}
        scans = {r["scan_id"]: (r["label"], r["window"]) for r in recs}
        n_ad = sum(l == CLASS_TO_INDEX["AD"] for l in subjects.values())
        print(f"{name}: {len(subjects)} subjects ({n_ad} AD / "
              f"{len(subjects) - n_ad} NC), {len(scans)} scans, {len(recs)} slices")
        windows = Counter((w, l) for l, w in scans.values())
        for (w, l), n in sorted(windows.items()):
            print(f"    window {w[0]}-{w[1]} {INDEX_TO_CLASS[l]}: {n} scans")


RAW_H, RAW_W = 240, 256 # size of every ADNI slice
PAD_SIZE = 256 # slices are zero-padded to PAD_SIZE x PAD_SIZE
_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "norm_stats.json")

def _pad_to_square(img):
    """
    Zero-pad a 1xHxW tensor to PAD_SIZE x PAD_SIZE (split evenly on each side).

    Zero matches the black MRI background, so padding adds no new structure.
    """
    _, h, w = img.shape
    ph, pw = PAD_SIZE - h, PAD_SIZE - w
    return TF.pad(img, [pw // 2, ph // 2, pw - pw // 2, ph - ph // 2], fill=0)


def _load_padded(path):
    """
    Load one JPEG as a padded float tensor in [0, 1], shape 1 x 256 x 256.
    """
    img = TF.to_tensor(Image.open(path).convert("L"))
    return _pad_to_square(img)

def compute_norm_stats(train_records, seed=SEED, stats_path=_STATS_PATH):
    """
    Mean and std of pixel intensity over the TRAIN split only.

    Using only training data keeps val/test information out of the preprocessing.
    The result is cached (small JSON) and reused if the seed and the number of
    training slices still match.
    """
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            cached = json.load(f)
        if cached["seed"] == seed and cached["n_images"] == len(train_records):
            return cached["mean"], cached["std"]

    total, total_sq, count = 0.0, 0.0, 0
    for r in train_records:
        x = _load_padded(r["path"]).double()
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()
        count += x.numel()
    mean = total / count
    std = (total_sq / count - mean ** 2) ** 0.5

    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"seed": seed, "n_images": len(train_records),
                   "mean": mean, "std": std}, f)
    return mean, std


def build_transforms(train, mean, std, img_size=PAD_SIZE, hflip=False):
    """
    Compose the preprocessing pipeline.

    Evaluation: pad -> (resize if img_size != 256) -> normalise.
    Training adds mild augmentation after padding, so rotations and shifts
    never crop the brain: small affine (rotation, translation, scale) and
    brightness/contrast jitter to mimic scanner intensity variation.
    Horizontal flip is off by default because slice or orientation is unverified.
    """
    steps = [transforms.Lambda(_pad_to_square)]
    if train:
        steps.append(transforms.RandomAffine(
        degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0))
        steps.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        if hflip:
            steps.append(transforms.RandomHorizontalFlip())

    if img_size != PAD_SIZE:
        steps.append(transforms.Resize((img_size, img_size), antialias=True))
    steps.append(transforms.Normalize([mean], [std]))
    return transforms.Compose(steps)


class ADNISliceDataset(Dataset):
    """
    One 2D slice per item: returns (image tensor 1 x S x S, label 0/1)

    The record list is kept in `self.records`, so predictions can later be
    traced back to subject, scan and window (order is preserved when the
    loader is not shuffled).
    """

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = TF.to_tensor(Image.open(r["path"]).convert("L"))
        return self.transform(img), r["label"]


def get_dataloaders(batch_size=64, img_size=PAD_SIZE, seed=SEED, num_workers=4,
                    hflip=False, root=ADNI_ROOT, subset=None):
    """
    Build train/val/test DataLoaders from the subject-level split.

    Returns a dict {'train', 'val', 'test'}. Only the train loader shuffles.
    """
    splits = split_subjects(collect_samples(root), seed=seed)
    mean, std = compute_norm_stats(splits["train"], seed=seed)

    if subset:  # smoke tests only: random subset of slices per split
        rng = random.Random(seed)
        splits = {k: rng.sample(v, min(subset, len(v))) for k, v in splits.items()}

    loaders = {}
    for name, records in splits.items():
        train = name == "train"
        dataset = ADNISliceDataset(
            records, build_transforms(train, mean, std, img_size, hflip))
        generator = torch.Generator().manual_seed(seed) if train else None
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True, generator=generator
        )
    return loaders