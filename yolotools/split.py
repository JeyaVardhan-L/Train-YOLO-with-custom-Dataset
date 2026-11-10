"""Split an image + label collection into train and validation folders, optionally by group."""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import count_by_class, group_key, list_images, read_labels


def split_dataset(
    source: str,
    output: str,
    val_pct: float = 0.2,
    seed: int = 42,
    group_regex: Optional[str] = None,
) -> Tuple[int, int]:
    """Copy images and labels into ``output/{train,val}/{images,labels}``.

    ``source`` is either a folder holding ``images/`` and ``labels/`` or one flat folder holding
    both. With ``group_regex`` all files in the same group (for example augmented copies of one
    photo) always land on the same side, which prevents train/validation leakage. Without it,
    every image is its own group and the split is random per image.

    Returns ``(train_images, val_images)``.
    """
    if not 0.01 <= val_pct <= 0.5:
        raise ValueError("val_pct must be between 0.01 and 0.5")
    src = Path(source)
    images_dir = src / "images" if (src / "images").is_dir() else src
    labels_dir = src / "labels" if (src / "labels").is_dir() else src
    images = list_images(images_dir)
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    groups: Dict[str, List[Path]] = {}
    for img in images:
        groups.setdefault(group_key(img.name, group_regex), []).append(img)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target_val = max(1, round(len(images) * val_pct))
    val_images: List[Path] = []
    train_images: List[Path] = []
    for key in keys:
        # Fill validation first until it reaches the target size; whole groups only.
        if len(val_images) < target_val:
            val_images.extend(groups[key])
        else:
            train_images.extend(groups[key])
    if not train_images:
        raise ValueError("Validation would take every image; lower val_pct or check the group pattern")

    out = Path(output)
    for split, files in (("train", train_images), ("val", val_images)):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, out / split / "images" / img.name)
            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, out / split / "labels" / label.name)
    return len(train_images), len(val_images)


def class_counts(split_dir: Path, num_classes: Optional[int] = None) -> Dict[int, int]:
    """Count boxes per class in ``split_dir/labels`` (used to show the result of a split)."""
    boxes = []
    for label in (Path(split_dir) / "labels").glob("*.txt"):
        boxes.extend(read_labels(label, num_classes)[0])
    return count_by_class(boxes)


def apply_split(source: str, output: str, train_list: str, val_list: str) -> Tuple[int, int, List[str]]:
    """Rebuild a known split: copy the images named in two list files (one file name per line).

    This recreates an existing train/validation split from the original, unsplit data, so that
    published results can be reproduced on exactly the same validation images.

    Returns ``(train_copied, val_copied, missing_names)``.
    """
    src = Path(source)
    images_dir = src / "images" if (src / "images").is_dir() else src
    labels_dir = src / "labels" if (src / "labels").is_dir() else src
    out = Path(output)
    missing: List[str] = []
    copied = {"train": 0, "val": 0}
    for split, list_file in (("train", train_list), ("val", val_list)):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        names = [n.strip() for n in Path(list_file).read_text(encoding="utf-8").splitlines() if n.strip()]
        for name in names:
            image = images_dir / name
            if not image.is_file():
                missing.append(name)
                continue
            shutil.copy2(image, out / split / "images" / name)
            label = labels_dir / f"{image.stem}.txt"
            if label.is_file():
                shutil.copy2(label, out / split / "labels" / label.name)
            copied[split] += 1
    return copied["train"], copied["val"], missing
