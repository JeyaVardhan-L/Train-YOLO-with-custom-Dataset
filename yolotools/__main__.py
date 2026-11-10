"""Command line entry: ``python -m yolotools <command> ...``."""

import argparse
import sys

from .common import ROBOFLOW_GROUP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m yolotools", description="Dataset tools for YOLO detection.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("yaml", help="write a data.yaml for a dataset folder")
    p.add_argument("--root", required=True, help="dataset folder holding train/ and val/")
    p.add_argument("--names", required=True, help="comma-separated class names, in class-id order")
    p.add_argument("--out", default="data.yaml")
    p.add_argument("--relative", action="store_true", help="write a relative path (resolved from the working directory)")

    p = sub.add_parser("check", help="check pairing, label format and class counts")
    p.add_argument("--data", required=True, help="path to data.yaml")

    p = sub.add_parser("split", help="split images + labels into train/ and val/")
    p.add_argument("--source", required=True, help="folder with images/ and labels/, or one flat folder")
    p.add_argument("--out", required=True)
    p.add_argument("--val-pct", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group", choices=["none", "roboflow"], default="none", help="keep files of one source photo together")
    p.add_argument("--group-regex", help="custom regex; first capture group is the group name (overrides --group)")

    p = sub.add_parser("apply-split", help="rebuild a known split from file lists (reproduce published results)")
    p.add_argument("--source", required=True, help="folder with all images (and labels), as downloaded")
    p.add_argument("--train-list", required=True, help="text file: one training image file name per line")
    p.add_argument("--val-list", required=True, help="text file: one validation image file name per line")
    p.add_argument("--out", required=True)

    p = sub.add_parser("leakage", help="count validation images that share a source photo with training images")
    p.add_argument("--train", required=True, help="training images folder")
    p.add_argument("--val", required=True, help="validation images folder")
    p.add_argument("--group", choices=["roboflow"], default="roboflow")
    p.add_argument("--group-regex", help="custom regex; first capture group is the group name")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "yaml":
        from .yamlgen import write_data_yaml

        path = write_data_yaml(args.root, [n.strip() for n in args.names.split(",")], out=args.out, relative=args.relative)
        print(f"Wrote {path}")
    elif args.command == "check":
        from .check import check_dataset

        return 1 if check_dataset(args.data) else 0
    elif args.command == "split":
        from .split import split_dataset

        regex = args.group_regex or (ROBOFLOW_GROUP if args.group == "roboflow" else None)
        train, val = split_dataset(args.source, args.out, args.val_pct, args.seed, regex)
        print(f"train: {train} images, val: {val} images -> {args.out}")
    elif args.command == "apply-split":
        from .split import apply_split

        train, val, missing = apply_split(args.source, args.out, args.train_list, args.val_list)
        print(f"train: {train} images, val: {val} images -> {args.out}")
        if missing:
            print(f"ERROR: {len(missing)} listed files were not found in {args.source}, for example: {missing[:3]}")
            return 1
    elif args.command == "leakage":
        from .leakage import find_leakage

        regex = args.group_regex or ROBOFLOW_GROUP
        r = find_leakage(args.train, args.val, regex)
        print(f"Validation images:             {r['val_images']}")
        print(f"Sharing a group with train:    {r['val_images_leaking']} ({r['leaking_percent']}%)")
        print(f"Distinct shared groups:        {r['shared_groups']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
