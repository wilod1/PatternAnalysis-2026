"""
Dataset utilities for ADNI Alzheimer's disease classification.

Each item is a 2D grayscale MRI slice. Images from the supplied train and
test directories are pooled and then split by subject, because the original
directories may contain slices from the same subject.

Labels:
    NC -> 0
    AD -> 1
"""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ADNI_ROOT = Path("/home/groups/comp3710/ADNI")
METADATA_FILENAME = "meta_data_with_label.json"

CLASS_TO_INDEX = {"NC": 0, "AD": 1}
JSON_LABEL_TO_CLASS = {0: "NC", 2: "AD"}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}

SEED = 42
IMAGE_SIZE = 256
RAW_HEIGHT = 240
RAW_WIDTH = 256

SUBJECT_PATTERN = re.compile(r"ADNI_(\d+_S_\d+)")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STATS_PATH = RESULTS_DIR / "norm_stats.json"

# (rotation, translation, scale, intensity_jitter, gamma, erasing_probability)
AUGMENTATIONS = {
    "mild": (8, 0.03, (0.95, 1.05), 0.10, None, 0.00),
    "strong": (15, 0.08, (0.85, 1.15), 0.30, (0.70, 1.50), 0.25),
}

# "strong_blur" is "strong" plus a random Gaussian blur.
AUGMENTATIONS["strong_blur"] = AUGMENTATIONS["strong"]

# preset -> (kernel_size, sigma range, probability); None means no blur.
BLUR = {
    "mild": None,
    "strong": None,
    "strong_blur": (5, (0.3, 1.2), 0.5),
}


# ---------------------------------------------------------------------
# Metadata and sample collection
# ---------------------------------------------------------------------

def load_scan_table(root=ADNI_ROOT):
    """Map each scan ID to its subject ID and original metadata label."""
    root = Path(root).expanduser().resolve()
    metadata_path = root / METADATA_FILENAME

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"ADNI metadata file was not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    scan_table = {}

    for scan_id, entry in metadata.items():
        match = SUBJECT_PATTERN.search(entry.get("raw", ""))

        if match is None:
            raise ValueError(
                f"Could not extract a subject ID for scan {scan_id}"
            )

        scan_table[str(scan_id)] = {
            "subject_id": match.group(1),
            "metadata_label": entry["label"],
        }

    return scan_table


def collect_samples(root=ADNI_ROOT):
    """
    Collect slice records from ADNI/AD_NC/{train,test}/{NC,AD}.

    Every record contains its image path, class label, scan ID, subject ID,
    slice index and scan-window boundaries.
    """
    root = Path(root).expanduser().resolve()
    scan_table = load_scan_table(root)
    records = []

    for supplied_split in ("train", "test"):
        for class_name, label in CLASS_TO_INDEX.items():
            folder = root / "AD_NC" / supplied_split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Expected image directory was not found: {folder}"
                )

            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() != ".jpeg":
                    continue

                parts = path.stem.rsplit("_", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Expected filename SCAN_ID_SLICE.jpeg: {path.name}"
                    )

                scan_id, slice_text = parts

                if scan_id not in scan_table:
                    raise KeyError(
                        f"Scan {scan_id} is missing from the metadata JSON"
                    )

                scan_info = scan_table[scan_id]
                metadata_label = scan_info["metadata_label"]

                if metadata_label not in JSON_LABEL_TO_CLASS:
                    raise ValueError(
                        f"Unexpected metadata label {metadata_label!r} "
                        f"for scan {scan_id}"
                    )

                expected_class = JSON_LABEL_TO_CLASS[metadata_label]

                if expected_class != class_name:
                    raise ValueError(
                        f"Class mismatch for scan {scan_id}: "
                        f"folder={class_name}, metadata={expected_class}"
                    )

                try:
                    slice_index = int(slice_text)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid slice index in filename: {path.name}"
                    ) from error

                records.append({
                    "path": str(path),
                    "label": label,
                    "scan_id": scan_id,
                    "subject_id": scan_info["subject_id"],
                    "slice_index": slice_index,
                })

    if not records:
        raise RuntimeError(
            f"No JPEG slices were found under {root / 'AD_NC'}"
        )

    # Determine the observed slice-index window for each scan.
    scan_windows = {}

    for record in records:
        scan_id = record["scan_id"]
        index = record["slice_index"]

        if scan_id not in scan_windows:
            scan_windows[scan_id] = [index, index]
        else:
            scan_windows[scan_id][0] = min(
                scan_windows[scan_id][0], index
            )
            scan_windows[scan_id][1] = max(
                scan_windows[scan_id][1], index
            )

    for record in records:
        record["window"] = tuple(scan_windows[record["scan_id"]])

    return records


# ---------------------------------------------------------------------
# Subject-level splitting and summaries
# ---------------------------------------------------------------------

def split_subjects(samples, seed=SEED, ratios=(0.70, 0.15, 0.15)):
    """
    Split subjects into train, validation and test sets.

    All scans and slices belonging to a subject remain in one split.
    The split is stratified by subject diagnosis and reproducible by seed.
    """
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")

    subject_labels = {}

    for record in samples:
        subject_id = record["subject_id"]
        label = record["label"]

        previous = subject_labels.setdefault(subject_id, label)

        if previous != label:
            raise ValueError(
                f"Subject {subject_id} has conflicting diagnosis labels"
            )

    rng = random.Random(seed)
    assignment = {}

    # Split each class independently to maintain approximate stratification.
    for label in sorted(set(subject_labels.values())):
        subjects = sorted(
            subject_id
            for subject_id, subject_label in subject_labels.items()
            if subject_label == label
        )
        rng.shuffle(subjects)

        n_subjects = len(subjects)
        n_train = round(ratios[0] * n_subjects)
        n_val = round(ratios[1] * n_subjects)

        # Keep all subjects assigned exactly once.
        if n_train + n_val > n_subjects:
            n_val = n_subjects - n_train

        for index, subject_id in enumerate(subjects):
            if index < n_train:
                assignment[subject_id] = "train"
            elif index < n_train + n_val:
                assignment[subject_id] = "val"
            else:
                assignment[subject_id] = "test"

    splits = {"train": [], "val": [], "test": []}

    for record in samples:
        split_name = assignment[record["subject_id"]]
        splits[split_name].append(record)

    # Explicitly verify that no subject occurs in more than one split.
    subject_sets = {
        name: {record["subject_id"] for record in records}
        for name, records in splits.items()
    }

    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["val"].isdisjoint(subject_sets["test"])

    return splits


def describe_splits(splits):
    """Print subject, scan, slice and class counts for each split."""
    for name, records in splits.items():
        subjects = {
            record["subject_id"]: record["label"]
            for record in records
        }

        scans = {
            record["scan_id"]: (
                record["label"],
                record["window"],
            )
            for record in records
        }

        n_ad = sum(label == CLASS_TO_INDEX["AD"]
                   for label in subjects.values())
        n_nc = len(subjects) - n_ad

        print(
            f"{name}: {len(subjects)} subjects "
            f"({n_nc} NC / {n_ad} AD), "
            f"{len(scans)} scans, {len(records)} slices"
        )

        window_counts = Counter(
            (window, label)
            for label, window in scans.values()
        )

        for (window, label), count in sorted(window_counts.items()):
            print(
                f"    window {window[0]}-{window[1]} "
                f"{INDEX_TO_CLASS[label]}: {count} scans"
            )


# ---------------------------------------------------------------------
# Image loading and normalisation
# ---------------------------------------------------------------------

def load_padded_image(path):
    """
    Load an MRI slice as a float tensor in [0, 1], shape (1, 256, 256).

    The original 240 x 256 image is padded vertically with black pixels.
    """
    with Image.open(path) as image:
        image_tensor = TF.to_tensor(image.convert("L"))

    _, height, width = image_tensor.shape

    if height > IMAGE_SIZE or width > IMAGE_SIZE:
        raise ValueError(
            f"Image dimensions {height} x {width} exceed the "
            f"configured {IMAGE_SIZE} x {IMAGE_SIZE} canvas: {path}"
        )

    pad_height = IMAGE_SIZE - height
    pad_width = IMAGE_SIZE - width

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    return TF.pad(
        image_tensor,
        [left, top, right, bottom],
        fill=0,
    )


def compute_norm_stats(train_records, seed=SEED, stats_path=STATS_PATH):
    """
    Compute grayscale mean and standard deviation using training images only.

    Cached statistics are reused only if the seed and training-record
    fingerprint match. Validation and test images never enter this calculation.
    """
    stats_path = Path(stats_path)

    fingerprint_source = "\n".join(
        f"{record['path']}|{record['label']}"
        for record in sorted(
            train_records,
            key=lambda item: item["path"],
        )
    )
    fingerprint = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()

    if stats_path.is_file():
        try:
            with stats_path.open("r", encoding="utf-8") as file:
                cached = json.load(file)

            if (
                cached.get("seed") == seed
                and cached.get("fingerprint") == fingerprint
            ):
                mean = float(cached["mean"])
                std = float(cached["std"])

                if std > 0:
                    print(
                        f"Using cached training normalisation: "
                        f"mean={mean:.6f}, std={std:.6f}"
                    )
                    return mean, std
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if not train_records:
        raise ValueError("Cannot calculate normalisation without training data")

    total = 0.0
    total_squared = 0.0
    pixel_count = 0

    for record in train_records:
        image = load_padded_image(record["path"]).double()

        total += image.sum().item()
        total_squared += image.square().sum().item()
        pixel_count += image.numel()

    mean = total / pixel_count
    variance = max(total_squared / pixel_count - mean * mean, 0.0)
    std = variance ** 0.5

    if std <= 0:
        raise ValueError(
            "Training image standard deviation is zero; "
            "check the image files and preprocessing."
        )

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = stats_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump({
            "seed": seed,
            "fingerprint": fingerprint,
            "n_images": len(train_records),
            "mean": mean,
            "std": std,
        }, file, indent=2)

    temporary_path.replace(stats_path)

    print(
        f"Computed training normalisation: "
        f"mean={mean:.6f}, std={std:.6f}"
    )

    return mean, std


class RandomGamma:
    """Apply a random gamma transformation to a grayscale tensor in [0, 1]."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        gamma = torch.empty(1).uniform_(
            self.low, self.high
        ).item()

        return image.clamp(0, 1).pow(gamma)


def build_transforms(
    training,
    mean,
    std,
    img_size=IMAGE_SIZE,
    aug="mild",
    horizontal_flip=False,
):
    """Build separate augmentation and evaluation pipelines."""
    if aug not in AUGMENTATIONS:
        raise ValueError(
            f"Unknown augmentation preset {aug!r}; "
            f"choose from {list(AUGMENTATIONS)}"
        )

    if img_size < 1:
        raise ValueError("img_size must be positive")

    steps = [transforms.Lambda(
        lambda image: _ensure_padded_size(image)
    )]

    if training:
        degrees, translation, scale, jitter, gamma, erase = AUGMENTATIONS[aug]

        steps.append(transforms.RandomAffine(
            degrees=degrees,
            translate=(translation, translation),
            scale=scale,
            fill=0,
        ))

        steps.append(transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
        ))

        blur = BLUR.get(aug)

        if blur is not None:
            kernel, sigma, probability = blur
            steps.append(transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)],
                p=probability,
            ))

        if gamma is not None:
            steps.append(RandomGamma(*gamma))

        if erase > 0:
            steps.append(transforms.RandomErasing(
                p=erase,
                scale=(0.01, 0.04),
                value=0,
            ))

        # Disabled by default until orientation suitability is established.
        if horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip(p=0.5))

    if img_size != IMAGE_SIZE:
        steps.append(transforms.Resize(
            (img_size, img_size),
            antialias=True,
        ))

    steps.append(transforms.Normalize([mean], [std]))

    return transforms.Compose(steps)


def _ensure_padded_size(image):
    """Pad a tensor image to the configured square canvas."""
    _, height, width = image.shape

    if height > IMAGE_SIZE or width > IMAGE_SIZE:
        raise ValueError(
            f"Image dimensions {height} x {width} exceed "
            f"{IMAGE_SIZE} x {IMAGE_SIZE}"
        )

    pad_height = IMAGE_SIZE - height
    pad_width = IMAGE_SIZE - width

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    return TF.pad(image, [left, top, right, bottom], fill=0)


# ---------------------------------------------------------------------
# Dataset and DataLoaders
# ---------------------------------------------------------------------

class ADNISliceDataset(Dataset):
    """Dataset returning (image tensor, integer diagnosis label)."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = load_padded_image(record["path"])
        image = self.transform(image)

        label = torch.tensor(record["label"], dtype=torch.long)
        return image, label


def _seed_worker(worker_id):
    """Seed Python and NumPy-compatible worker RNGs from PyTorch's seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)

    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


def get_dataloaders(
    batch_size=64,
    img_size=IMAGE_SIZE,
    seed=SEED,
    num_workers=4,
    horizontal_flip=False,
    root=ADNI_ROOT,
    subset=None,
    aug="mild",
    pin_memory=None,
):
    """
    Return a dictionary containing 'train', 'val' and 'test' DataLoaders.

    `subset` is intended only for smoke tests, not reported experiments.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    samples = collect_samples(root)
    splits = split_subjects(samples, seed=seed)

    print("\nDataset split summary")
    describe_splits(splits)

    # Calculate statistics from the full training split before optional
    # smoke-test subsampling, so the cache describes the actual split.
    mean, std = compute_norm_stats(splits["train"], seed=seed)

    if subset is not None:
        if subset < 1:
            raise ValueError("subset must be positive when specified")

        rng = random.Random(seed)
        splits = {
            name: rng.sample(records, min(subset, len(records)))
            for name, records in splits.items()
        }

        print(
            "\nWARNING: subset mode is enabled. "
            "Use this only for smoke tests."
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {}

    for split_name, records in splits.items():
        training = split_name == "train"

        dataset = ADNISliceDataset(
            records=records,
            transform=build_transforms(
                training=training,
                mean=mean,
                std=std,
                img_size=img_size,
                aug=aug,
                horizontal_flip=horizontal_flip,
            ),
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            drop_last=training and len(dataset) > batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    return loaders