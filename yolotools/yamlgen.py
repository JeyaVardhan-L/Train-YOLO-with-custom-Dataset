"""Write a data.yaml that Ultralytics can read from any working directory."""

from pathlib import Path
from typing import List

import yaml


def write_data_yaml(dataset_root: str, names: List[str], out: str = "data.yaml", train: str = "train/images", val: str = "val/images", relative: bool = False) -> Path:
    """Create ``out`` describing a dataset folder.

    By default ``path`` is written as an absolute path, so training works no matter where you run
    it. Ultralytics resolves a *relative* ``path`` against the current working directory, not the
    folder that holds the YAML, which is a common cause of "images not found" errors.
    """
    root = Path(dataset_root)
    path_value = str(root) if relative else str(root.resolve())
    config = {
        "path": path_value,
        "train": train,
        "val": val,
        "names": {i: name for i, name in enumerate(names)},
    }
    out_path = Path(out)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out_path
