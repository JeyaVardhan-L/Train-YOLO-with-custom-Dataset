"""Check a YOLO detection dataset described by a data.yaml: pairing, label format, class counts."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from .common import count_by_class, labels_dir_for, list_images, read_labels


def load_data_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load a data.yaml and return its split image folders and class names, resolved to paths.

    Ultralytics resolves a relative ``path`` against the *current working directory* (not the
    folder holding the YAML), so this function does the same.
    """
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    for key in ("train", "val", "names"):
        if key not in cfg:
            raise ValueError(f"data.yaml is missing the '{key}:' key")
    root = Path(cfg.get("path", "."))
    names = cfg["names"]
    names = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}
    return {
        "root": root,
        "names": names,
        "splits": {"train": (root / cfg["train"]).resolve(), "val": (root / cfg["val"]).resolve()},
    }


def check_split(images_dir: Path, num_classes: int) -> Dict[str, Any]:
    """Inspect one split and return counts plus lists of problems."""
    result: Dict[str, Any] = {
        "images_dir": str(images_dir),
        "images": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "empty_labels": 0,
        "invalid_lines": [],
        "class_counts": {},
        "boxes": 0,
        "tiny_boxes": 0,
    }
    if not images_dir.is_dir():
        result["error"] = "images folder not found"
        return result
    labels_dir = labels_dir_for(images_dir)
    if not labels_dir.is_dir():
        result["error"] = f"labels folder not found: {labels_dir}"
        return result

    images = list_images(images_dir)
    result["images"] = len(images)
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}
    all_boxes: List = []
    for img in images:
        label = label_files.pop(img.stem, None)
        if label is None:
            result["missing_labels"].append(img.name)
            continue
        boxes, problems = read_labels(label, num_classes)
        result["invalid_lines"].extend(problems)
        if not boxes and not problems:
            result["empty_labels"] += 1
        all_boxes.extend(boxes)
    result["orphan_labels"] = sorted(f"{stem}.txt" for stem in label_files)
    result["class_counts"] = count_by_class(all_boxes)
    result["boxes"] = len(all_boxes)
    # Boxes covering under 0.1% of the image are very hard for a detector at 640 px.
    result["tiny_boxes"] = sum(1 for b in all_boxes if b[3] * b[4] < 0.001)
    return result


def check_dataset(yaml_path: str, show: int = 5) -> int:
    """Print a report for every split. Returns the number of problems found (0 means clean)."""
    cfg = load_data_yaml(yaml_path)
    names = cfg["names"]
    problems = 0
    print(f"Classes ({len(names)}): {names}")
    for split, images_dir in cfg["splits"].items():
        r = check_split(images_dir, len(names))
        print(f"\n[{split}] {r['images_dir']}")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            problems += 1
            continue
        print(f"  images: {r['images']}   boxes: {r['boxes']}   background images (empty label): {r['empty_labels']}")
        for cid, name in names.items():
            print(f"    class {cid} ({name}): {r['class_counts'].get(cid, 0)}")
        absent = [n for c, n in names.items() if c not in r["class_counts"]]
        if absent:
            print(f"  WARNING: no examples of {absent} in this split")
        if r["tiny_boxes"]:
            n = r["tiny_boxes"]
            print(f"  note: {n} {'box covers' if n == 1 else 'boxes cover'} under 0.1% of the image (hard to detect)")
        for label, items in (
            ("images without a label file", r["missing_labels"]),
            ("label files without an image", r["orphan_labels"]),
            ("invalid label lines", r["invalid_lines"]),
        ):
            if items:
                problems += len(items)
                print(f"  ERROR: {len(items)} {label}, for example:")
                for item in items[:show]:
                    print(f"    - {item}")
    print("\nResult:", "no problems found" if problems == 0 else f"{problems} problem(s) found")
    return problems
