"""Small helpers shared by the dataset tools: file discovery, label parsing, grouping."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Filenames exported by Roboflow look like "<source>_jpg.rf.<hash>.jpg". Augmented copies of one
# source photo share <source> but get different hashes.
ROBOFLOW_GROUP = r"^(.+?)[._-]rf[._-]"

Box = Tuple[int, float, float, float, float]


def list_images(directory: Path) -> List[Path]:
    """Image files directly inside ``directory`` (sorted, case-insensitive extensions)."""
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def labels_dir_for(images_dir: Path) -> Path:
    """Ultralytics rule: the labels folder is the images path with the last ``images`` part swapped."""
    parts = list(Path(images_dir).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise ValueError(f"Cannot derive a labels folder: no 'images' folder in {images_dir}")


def group_key(filename: str, pattern: Optional[str]) -> str:
    """Return the group a file belongs to.

    With no pattern every file is its own group. With a regex, the first capture group of a match
    (or the whole match when there is no capture group) is the key; files that do not match are
    their own group.
    """
    name = Path(filename).name
    if not pattern:
        return name
    match = re.match(pattern, name)
    if not match:
        return name
    return match.group(1) if match.groups() else match.group(0)


def parse_label_line(line: str, num_classes: Optional[int]) -> Tuple[Optional[Box], Optional[str]]:
    """Parse one YOLO detection label line; return ``(box, None)`` or ``(None, problem)``."""
    parts = line.split()
    if len(parts) != 5:
        return None, f"expected 5 fields, found {len(parts)}"
    try:
        cls = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None, "non-numeric value"
    if cls < 0 or (num_classes is not None and cls >= num_classes):
        return None, f"class id {cls} is outside the configured classes"
    if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
        return None, "coordinate outside [0, 1] (labels must be normalised)"
    if w <= 0 or h <= 0:
        return None, "box with zero width or height"
    return (cls, x, y, w, h), None


def read_labels(path: Path, num_classes: Optional[int]) -> Tuple[List[Box], List[str]]:
    """Read a label file; return the valid boxes and a list of problem descriptions."""
    boxes: List[Box] = []
    problems: List[str] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        box, problem = parse_label_line(line, num_classes)
        if box:
            boxes.append(box)
        else:
            problems.append(f"{Path(path).name}:{number}: {problem}")
    return boxes, problems


def count_by_class(boxes: List[Box]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for box in boxes:
        counts[box[0]] = counts.get(box[0], 0) + 1
    return dict(sorted(counts.items()))
