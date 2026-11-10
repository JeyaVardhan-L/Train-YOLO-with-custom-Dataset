"""Measure how many validation images share a source photo with a training image."""

from pathlib import Path
from typing import Dict

from .common import group_key, list_images


def find_leakage(train_images: str, val_images: str, group_regex: str) -> Dict[str, object]:
    """Count validation images whose group (see ``group_key``) also occurs in the training images."""
    train_groups = {group_key(p.name, group_regex) for p in list_images(Path(train_images))}
    val_files = list_images(Path(val_images))
    leaking = [p for p in val_files if group_key(p.name, group_regex) in train_groups]
    total = len(val_files)
    return {
        "val_images": total,
        "val_images_leaking": len(leaking),
        "leaking_percent": round(100.0 * len(leaking) / total, 1) if total else 0.0,
        "shared_groups": len({group_key(p.name, group_regex) for p in leaking}),
    }
