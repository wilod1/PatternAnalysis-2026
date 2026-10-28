"""
Dataset utilities for ADNI Alzheimer's disease classification.

Each item is a 2D grayscale MRI slice. Images from the supplied train and
test directories are pooled and then split by subject, because the original
directories may contain slices from the same subject.

Labels:
    NC -> 0
    AD -> 1

Two evaluation protocols are supported (chosen with split_mode in
get_dataloaders):
    "subject"   patient-disjoint development split of the pooled scans
                (70/15/15 by patient, stratified by diagnosis, fixed seed)
    "provided"  the supplied train and test folders, with the train folder
                divided into train and validation by scan (kept for the
                headline comparison of the report; it can leak patients)

Main entry points
    collect_samples     read the metadata and list every slice as a record
    split_subjects      patient-disjoint split
    split_provided      split of the supplied folders
    compute_norm_stats  mean and standard deviation from training images only
    build_transforms    padded 256 x 256 pipeline (mild / strong presets)
    build_stretch_transforms   resize-and-crop pipeline (224 x 224)
    get_dataloaders     build the train / val / test DataLoaders

This module is imported by train.py, utils.py and the analysis scripts; it is
not run directly.
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

# Course copy of the data on Rangpur; the images are read in place, never copied.
ADNI_ROOT = Path("/home/groups/comp3710/ADNI")
METADATA_FILENAME = "meta_data_with_label.json"

CLASS_TO_INDEX = {"NC": 0, "AD": 1}
# Diagnosis codes used in the metadata file: 0 is NC and 2 is AD. Any other code
# is rejected when the samples are collected.
JSON_LABEL_TO_CLASS = {0: "NC", 2: "AD"}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}

SEED = 42  # one seed fixes the splits, the sampling and the augmentations
IMAGE_SIZE = 256  # side of the square canvas the slices are padded to
RAW_HEIGHT = 240  # native size of every slice in the dataset
RAW_WIDTH = 256

# Patient identifier inside the raw path stored in the metadata, for example
# ADNI_<site>_S_<id>. The number at the start of a file name is a scan ID, not
# a patient ID, so this pattern is what identifies the patient.
SUBJECT_PATTERN = re.compile(r"ADNI_(\d+_S_\d+)")

# Normalisation statistics are cached next to this file, in results/
RESULTS_DIR = Path(__file__).resolve().parent / "results"
STATS_PATH = RESULTS_DIR / "norm_stats.json"

# Augmentation presets for the padded pipeline, as
# (rotation in degrees, translation as a fraction of the image,
#  scale range, brightness/contrast jitter, gamma range or None,
#  probability of random erasing)
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

# A separate resize-and-crop pipeline. Images are loaded at their native
# 240 x 256 size and stretched (not padded) to a square, then cropped at
# random for training. It is selected with aug="stretch_crop" and uses its
# own normalisation statistics, computed from the training split only.
STRETCH_CROP = "stretch_crop"
AUG_CHOICES = sorted(AUGMENTATIONS) + [STRETCH_CROP]


# ---------------------------------------------------------------------
# Metadata and sample collection
# ---------------------------------------------------------------------

def load_scan_table(root=ADNI_ROOT):
    """Map each scan ID to its subject ID and original metadata label."""
    root = Path(root).expanduser().resolve()
    metadata_path = root / METADATA_FILENAME

    # fail early with a clear message if the dataset is not where it is expected
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"ADNI metadata file was not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    scan_table = {}

    for scan_id, entry in metadata.items():
        # the patient identifier is read from the raw path of the scan
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

    # Both supplied folders are read; the original folder of each slice stays
    # recoverable from its path (see split_provided).
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

                # File names are SCANID_SLICE.jpeg, so split once from the right
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

                # Cross-check the folder against the metadata: a slice filed
                # under NC must belong to a scan labelled NC in the metadata.
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

    # Determine the observed slice-index window for each scan: the lowest and
    # highest slice index present. The window turns out to carry class
    # information, which is why a window-only baseline is reported.
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

def split_subjects(samples, seed=SEED, ratios=(0.70, 0.15, 0.15),
                   group_by="subject_id"):
    """
    Split subjects into train, validation and test sets.

    All scans and slices belonging to a subject remain in one split.
    The split is stratified by subject diagnosis and reproducible by seed.

    group_by="scan_id" groups by scan instead. That LEAKS: scans of one
    patient can land in different splits. It exists only to reproduce a
    scan-level split for a labelled comparison, never for reported results.
    """
    if group_by not in ("subject_id", "scan_id"):
        raise ValueError("group_by must be 'subject_id' or 'scan_id'")
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")

    # One diagnosis per group. A patient with scans of both classes would make
    # the stratified split ill-defined, so that case raises an error (the real
    # data has no such patient).
    subject_labels = {}

    for record in samples:
        subject_id = record[group_by]
        label = record["label"]

        previous = subject_labels.setdefault(subject_id, label)

        if previous != label:
            raise ValueError(
                f"Subject {subject_id} has conflicting diagnosis labels"
            )

    rng = random.Random(seed)  # local generator: the split never touches global state
    assignment = {}  # group id -> "train", "val" or "test"

    # Split each class independently to maintain approximate stratification.
    for label in sorted(set(subject_labels.values())):
        # sort first so the shuffle depends only on the seed, not on file order
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

        # first n_train groups -> train, next n_val -> val, the rest -> test
        for index, subject_id in enumerate(subjects):
            if index < n_train:
                assignment[subject_id] = "train"
            elif index < n_train + n_val:
                assignment[subject_id] = "val"
            else:
                assignment[subject_id] = "test"

    # Every slice follows its group, so all slices of a patient stay together.
    splits = {"train": [], "val": [], "test": []}

    for record in samples:
        split_name = assignment[record[group_by]]
        splits[split_name].append(record)

    # Explicitly verify that no group occurs in more than one split.
    subject_sets = {
        name: {record[group_by] for record in records}
        for name, records in splits.items()
    }

    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["val"].isdisjoint(subject_sets["test"])

    # For the leaky scan-level option, measure how many real patients are shared.
    if group_by != "subject_id":
        train_subjects = {r["subject_id"] for r in splits["train"]}
        for name in ("val", "test"):
            held = {r["subject_id"] for r in splits[name]}
            print(f"WARNING (leaky {group_by} split): {name} has "
                  f"{len(held)} subjects, {len(held & train_subjects)} "
                  f"also in train")

    return splits


def split_provided(samples, seed=SEED, val_fraction=0.15):
    """
    The course-staff protocol, kept for a labelled comparison.

    Test is the supplied test folder. The supplied train folder is split into
    train and validation by the number at the start of the file name (the scan
    ID), stratified by class. Because the metadata shows that one patient owns
    several scan IDs, this protocol can place a patient in more than one split;
    the overlap by metadata subject ID is printed so the leak is measured.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")

    def folder_of(record):
        """Supplied folder ("train" or "test") that a slice came from."""
        # path layout: .../AD_NC/<train|test>/<NC|AD>/<file>.jpeg
        return Path(record["path"]).parent.parent.name

    pool = [r for r in samples if folder_of(r) == "train"]
    test = [r for r in samples if folder_of(r) == "test"]

    if not pool or not test:
        raise RuntimeError("Expected both a train and a test folder")

    # Group by scan ID, as the supplied protocol does, and check that each scan
    # has a single label.
    group_labels = {}
    for record in pool:
        previous = group_labels.setdefault(record["scan_id"], record["label"])
        if previous != record["label"]:
            raise ValueError(f"Scan {record['scan_id']} has conflicting labels")

    rng = random.Random(seed)
    val_groups = set()  # scans held out for validation

    # Take val_fraction of the scans of each class (stratified by diagnosis)
    for label in sorted(set(group_labels.values())):
        groups = sorted(g for g, l in group_labels.items() if l == label)
        rng.shuffle(groups)
        n_val = round(val_fraction * len(groups))
        val_groups.update(groups[:n_val])

    splits = {
        "train": [r for r in pool if r["scan_id"] not in val_groups],
        "val": [r for r in pool if r["scan_id"] in val_groups],
        "test": test,
    }

    # No scan may appear in two splits (patients still can; that is the leak).
    train_scans = {r["scan_id"] for r in splits["train"]}
    assert train_scans.isdisjoint({r["scan_id"] for r in splits["val"]})
    assert train_scans.isdisjoint({r["scan_id"] for r in splits["test"]})

    # Measure the patient overlap using the patient IDs from the metadata
    train_subjects = {r["subject_id"] for r in splits["train"]}
    val_subjects = {r["subject_id"] for r in splits["val"]}
    test_subjects = {r["subject_id"] for r in splits["test"]}
    print(
        "NOTE (provided protocol): by metadata subject ID, "
        f"{len(val_subjects & train_subjects)}/{len(val_subjects)} val subjects "
        f"and {len((test_subjects) & (train_subjects | val_subjects))}/"
        f"{len(test_subjects)} test subjects also appear in train/val"
    )
    return splits


def describe_splits(splits):
    """Print subject, scan, slice and class counts for each split."""
    for name, records in splits.items():
        # dictionaries keyed by patient / scan remove duplicates, so len()
        # counts distinct patients and distinct scans
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

        # number of scans per (slice window, class), to show how the windows
        # differ in class composition
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
        # convert("L") gives one grayscale channel; to_tensor scales to [0, 1]
        image_tensor = TF.to_tensor(image.convert("L"))

    _, height, width = image_tensor.shape

    if height > IMAGE_SIZE or width > IMAGE_SIZE:
        raise ValueError(
            f"Image dimensions {height} x {width} exceed the "
            f"configured {IMAGE_SIZE} x {IMAGE_SIZE} canvas: {path}"
        )

    # Centre the image on the canvas: split the missing pixels between the two
    # sides (the extra pixel, if odd, goes to the right / bottom).
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


def load_raw_image(path):
    """Load an MRI slice at its native size as a float tensor in [0, 1]."""
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("L"))


def compute_norm_stats(
    train_records,
    seed=SEED,
    stats_path=STATS_PATH,
    *,
    mode="pad",
    img_size=IMAGE_SIZE,
):
    """
    Compute grayscale mean and standard deviation using training images only.

    mode="pad" measures the zero-padded 256 x 256 images (the default).
    mode="stretch" measures native images resized to img_size x img_size and
    caches to its own file, so the padded statistics are never overwritten.

    Cached statistics are reused only if the seed and training-record
    fingerprint match. Validation and test images never enter this calculation.
    """
    stats_path = Path(stats_path)

    # The resized images have different statistics, so they get their own file
    if mode == "stretch" and stats_path == STATS_PATH:
        stats_path = STATS_PATH.with_name(
            f"norm_stats_stretch{img_size}.json"
        )

    # Fingerprint of the training set: a hash of the sorted file paths and labels.
    # If the training records change, the hash changes and the cache is ignored.
    fingerprint_source = "\n".join(
        f"{record['path']}|{record['label']}"
        for record in sorted(
            train_records,
            key=lambda item: item["path"],
        )
    )

    # The padded fingerprint is left unchanged so its existing cache stays valid.
    if mode == "stretch":
        fingerprint_source = f"stretch|{img_size}\n" + fingerprint_source
    fingerprint = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()

    # Reuse the cached values when they were computed for exactly this seed and
    # training set; any problem reading the cache simply falls through to a
    # fresh computation.
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

    # Streaming computation: accumulate the sum and the sum of squares over every
    # pixel of every training image, so the images never have to fit in memory
    # at once. Training images only: no validation or test pixel is used.
    total = 0.0
    total_squared = 0.0
    pixel_count = 0

    for record in train_records:
        if mode == "stretch":
            image = TF.resize(
                load_raw_image(record["path"]),
                [img_size, img_size],
                antialias=True,
            ).double()  # double precision keeps the large sums accurate
        else:
            image = load_padded_image(record["path"]).double()

        total += image.sum().item()
        total_squared += image.square().sum().item()
        pixel_count += image.numel()

    # variance = E[x^2] - (E[x])^2; max() guards against a tiny negative value
    # caused by rounding
    mean = total / pixel_count
    variance = max(total_squared / pixel_count - mean * mean, 0.0)
    std = variance ** 0.5

    if std <= 0:
        raise ValueError(
            "Training image standard deviation is zero; "
            "check the image files and preprocessing."
        )

    # Write to a temporary file and then rename it, so a job that is interrupted
    # mid-write can never leave a half-written cache behind.
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
        """Gamma is drawn uniformly from [low, high] for every image."""
        self.low = low
        self.high = high

    def __call__(self, image):
        """Return image ** gamma; gamma < 1 brightens, gamma > 1 darkens."""
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

    # Every image is first padded to the square canvas (a no-op if already square)
    steps = [transforms.Lambda(
        lambda image: _ensure_padded_size(image)
    )]

    # Random augmentation is applied to training images only; validation and
    # test images get the padding and normalisation and nothing else.
    if training:
        degrees, translation, scale, jitter, gamma, erase = AUGMENTATIONS[aug]

        # random rotation, shift and zoom (empty corners are filled with black)
        steps.append(transforms.RandomAffine(
            degrees=degrees,
            translate=(translation, translation),
            scale=scale,
            fill=0,
        ))

        # random brightness and contrast change
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

        # optional random gamma change (strong presets only)
        if gamma is not None:
            steps.append(RandomGamma(*gamma))

        # optional small random black rectangle (strong presets only)
        if erase > 0:
            steps.append(transforms.RandomErasing(
                p=erase,
                scale=(0.01, 0.04),
                value=0,
            ))

        # Disabled by default until orientation suitability is established.
        if horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip(p=0.5))

    # Resize only when a size other than the 256 canvas is requested
    if img_size != IMAGE_SIZE:
        steps.append(transforms.Resize(
            (img_size, img_size),
            antialias=True,
        ))

    # Standardise with the mean and standard deviation of the training images
    steps.append(transforms.Normalize([mean], [std]))

    return transforms.Compose(steps)


def build_stretch_transforms(training, mean, std, img_size=224):
    """
    Resize-and-crop pipeline for native 240 x 256 tensors.

    Training: stretch to a slightly larger square, random-resized crop to
    img_size, horizontal flip, affine jitter, brightness/contrast jitter and a
    light blur. Evaluation: a single resize to img_size x img_size.
    """
    if img_size < 32:
        raise ValueError("img_size must be at least 32")

    if training:
        # stretch to a slightly larger square (256 for a 224 target), then crop
        pre_size = round(img_size * 256 / 224)

        steps = [
            transforms.Resize((pre_size, pre_size), antialias=True),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.0),
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    else:
        # evaluation: one deterministic resize, no crop or augmentation
        steps = [
            transforms.Resize((img_size, img_size), antialias=True),
        ]

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

    def __init__(self, records, transform, loader=load_padded_image):
        """
        records:   list of slice records from collect_samples
        transform: pipeline applied to every loaded image
        loader:    function that reads an image file into a tensor
        """
        self.records = records
        self.transform = transform
        self.loader = loader

    def __len__(self):
        """Number of slices in this split."""
        return len(self.records)

    def __getitem__(self, index):
        """Load one slice, transform it, and return (image, label)."""
        record = self.records[index]
        image = self.loader(record["path"])
        image = self.transform(image)

        label = torch.tensor(record["label"], dtype=torch.long)
        return image, label


def _seed_worker(worker_id):
    """Seed Python and NumPy-compatible worker RNGs from PyTorch's seed."""
    # Each loader worker process gets its own seed derived from PyTorch's, so
    # augmentations differ between workers but repeat across runs.
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
    split_mode="subject",
):
    """
    Return a dictionary containing 'train', 'val' and 'test' DataLoaders.

    split_mode  "subject" (patient-disjoint split) or "provided" (supplied
                folders; train split into train and validation by scan)
    aug         padded-pipeline preset ("mild", "strong", "strong_blur") or
                "stretch_crop" for the resize-and-crop pipeline
    img_size    side of the network input in pixels
    `subset` is intended only for smoke tests, not reported experiments.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    # read every slice record, then split them according to the chosen protocol
    samples = collect_samples(root)
    if split_mode == "subject":
        splits = split_subjects(samples, seed=seed)
        stats_path = STATS_PATH
    elif split_mode == "provided":
        # Comparison only: supplied train/test folders, train split by scan.
        splits = split_provided(samples, seed=seed)
        stats_path = STATS_PATH.with_name("norm_stats_provided.json")
    else:
        raise ValueError("split_mode must be 'subject' or 'provided'")

    print("\nDataset split summary")
    describe_splits(splits)

    # Calculate statistics from the full training split before optional
    # smoke-test subsampling, so the cache describes the actual split.
    stretch = aug == STRETCH_CROP

    mean, std = compute_norm_stats(
        splits["train"],
        seed=seed,
        stats_path=stats_path,
        mode="stretch" if stretch else "pad",
        img_size=img_size,
    )

    # Smoke-test mode: keep a small random sample of every split
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

    # pinned memory speeds up host-to-GPU copies, so enable it when a GPU exists
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {}

    for split_name, records in splits.items():
        training = split_name == "train"

        # choose the pipeline and the matching image reader: the resize-and-crop
        # pipeline needs native-size images, the padded pipeline needs padded ones
        if stretch:
            transform = build_stretch_transforms(
                training, mean, std, img_size
            )
            image_loader = load_raw_image
        else:
            transform = build_transforms(
                training=training,
                mean=mean,
                std=std,
                img_size=img_size,
                aug=aug,
                horizontal_flip=horizontal_flip,
            )
            image_loader = load_padded_image

        dataset = ADNISliceDataset(
            records=records,
            transform=transform,
            loader=image_loader,
        )

        # seeded generator so the shuffling order is reproducible
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Only the training loader shuffles. Validation and test loaders keep the
        # file order, so their predictions line up with dataset.records. The last
        # incomplete training batch is dropped to keep batch statistics stable.
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