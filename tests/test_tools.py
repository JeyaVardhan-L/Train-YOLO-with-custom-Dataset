"""Tests for the dataset tools (run: python -m unittest discover -s tests -v)."""

import tempfile
import unittest
from pathlib import Path

import yaml

from yolotools.check import check_dataset, load_data_yaml
from yolotools.common import ROBOFLOW_GROUP, group_key, labels_dir_for, parse_label_line
from yolotools.leakage import find_leakage
from yolotools.split import apply_split, split_dataset
from yolotools.yamlgen import write_data_yaml


def make_dataset(root: Path, train: int = 12, val: int = 4) -> Path:
    """Tiny valid dataset: placeholder image files plus one label line each; returns the data.yaml path."""
    for split, count in (("train", train), ("val", val)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(count):
            name = f"{split}_{i:03d}"
            (root / split / "images" / f"{name}.jpg").write_bytes(bytes([255, 216, 255, 224]) + bytes(16))
            (root / split / "labels" / f"{name}.txt").write_text(f"{i % 3} 0.5 0.5 0.2 0.2" + chr(10), encoding="utf-8")
    return write_data_yaml(str(root), ["a", "b", "c"], out=str(root / "data.yaml"))


def make_grouped_collection(root: Path, groups: int = 20, copies: int = 3) -> None:
    """Flat folder of Roboflow-style names: ``photo<g>_jpg.rf.<hash>.jpg`` with ``copies`` per photo."""
    for g in range(groups):
        for c in range(copies):
            stem = f"photo{g:03d}_jpg.rf.hash{c}"
            (root / f"{stem}.jpg").write_bytes(bytes([255, 216, 255, 224]) + bytes(16))
            (root / f"{stem}.txt").write_text(f"{g % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


class TestLabelParsing(unittest.TestCase):
    def test_valid_line(self):
        self.assertEqual(parse_label_line("1 0.5 0.4 0.2 0.3", 3), ((1, 0.5, 0.4, 0.2, 0.3), None))

    def test_invalid_lines(self):
        for line, text in [
            ("1 0.5 0.4 0.2", "5 fields"),
            ("a 0.5 0.4 0.2 0.3", "non-numeric"),
            ("3 0.5 0.4 0.2 0.3", "class id"),
            ("1 1.5 0.4 0.2 0.3", "normalised"),
            ("1 0.5 0.4 0.0 0.3", "zero width"),
        ]:
            box, problem = parse_label_line(line, 3)
            self.assertIsNone(box, line)
            self.assertIn(text, problem, line)

    def test_labels_dir_for(self):
        self.assertEqual(labels_dir_for(Path("data/train/images")), Path("data/train/labels"))
        with self.assertRaises(ValueError):
            labels_dir_for(Path("data/train/pictures"))

    def test_group_key(self):
        self.assertEqual(group_key("photo001_jpg.rf.abc.jpg", ROBOFLOW_GROUP), "photo001_jpg")
        self.assertEqual(group_key("plain.jpg", ROBOFLOW_GROUP), "plain.jpg")
        self.assertEqual(group_key("a.jpg", None), "a.jpg")


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.yaml = make_dataset(self.root / "ds")

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_dataset(self):
        self.assertEqual(check_dataset(str(self.yaml)), 0)

    def test_detects_missing_label_bad_line_and_orphan(self):
        train = self.root / "ds" / "train"
        (train / "labels" / "train_000.txt").unlink()
        (train / "labels" / "train_001.txt").write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (train / "labels" / "stray.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        self.assertEqual(check_dataset(str(self.yaml)), 3)

    def test_missing_labels_folder_is_an_error(self):
        (self.root / "empty" / "train" / "images").mkdir(parents=True)
        (self.root / "empty" / "val" / "images").mkdir(parents=True)
        yaml_path = write_data_yaml(str(self.root / "empty"), ["a"], out=str(self.root / "empty.yaml"))
        self.assertGreater(check_dataset(str(yaml_path)), 0)


class TestSplitAndLeakage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "raw"
        self.src.mkdir()
        make_grouped_collection(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def _leak(self, out: Path):
        return find_leakage(str(out / "train" / "images"), str(out / "val" / "images"), ROBOFLOW_GROUP)

    def test_random_split_leaks_grouped_copies(self):
        out = self.root / "random"
        train, val = split_dataset(str(self.src), str(out), val_pct=0.2, seed=1)
        self.assertEqual(train + val, 60)
        self.assertGreater(self._leak(out)["val_images_leaking"], 0)

    def test_group_split_has_no_leakage(self):
        out = self.root / "grouped"
        train, val = split_dataset(str(self.src), str(out), val_pct=0.2, seed=1, group_regex=ROBOFLOW_GROUP)
        self.assertEqual(train + val, 60)
        self.assertEqual(self._leak(out)["val_images_leaking"], 0)
        self.assertGreaterEqual(val, 12)  # about 20% of 60, in whole groups of 3
        # labels travel with their images
        self.assertEqual(len(list((out / "val" / "labels").glob("*.txt"))), val)

    def test_split_is_deterministic(self):
        a, b = self.root / "a", self.root / "b"
        split_dataset(str(self.src), str(a), seed=7, group_regex=ROBOFLOW_GROUP)
        split_dataset(str(self.src), str(b), seed=7, group_regex=ROBOFLOW_GROUP)
        self.assertEqual(sorted(p.name for p in (a / "val" / "images").iterdir()), sorted(p.name for p in (b / "val" / "images").iterdir()))

    def test_bad_fraction(self):
        with self.assertRaises(ValueError):
            split_dataset(str(self.src), str(self.root / "x"), val_pct=0.9)


class TestApplySplit(unittest.TestCase):
    def test_rebuilds_listed_split_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "raw"
            src.mkdir()
            make_grouped_collection(src, groups=4, copies=2)
            names = sorted(p.name for p in src.glob("*.jpg"))
            (root / "train.txt").write_text(chr(10).join(names[:6] + ["not_there.jpg"]), encoding="utf-8")
            (root / "val.txt").write_text(chr(10).join(names[6:]), encoding="utf-8")
            train, val, missing = apply_split(str(src), str(root / "out"), str(root / "train.txt"), str(root / "val.txt"))
            self.assertEqual((train, val), (6, 2))
            self.assertEqual(missing, ["not_there.jpg"])
            self.assertEqual(len(list((root / "out" / "val" / "labels").glob("*.txt"))), 2)


class TestYaml(unittest.TestCase):
    def test_absolute_path_by_default_and_names_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_data_yaml(tmp, ["cat", "dog"], out=str(Path(tmp) / "data.yaml"))
            cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertTrue(Path(cfg["path"]).is_absolute())
            self.assertEqual(cfg["names"], {0: "cat", 1: "dog"})
            self.assertEqual(load_data_yaml(str(out))["names"], {0: "cat", 1: "dog"})


if __name__ == "__main__":
    unittest.main()
