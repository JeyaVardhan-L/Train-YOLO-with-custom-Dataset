#!/usr/bin/env python3
"""CLI Script to validate and evaluate trained YOLO checkpoints against validation split."""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on custom airborne object validation split."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights (.pt file) (default: weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data.yaml",
        help="Path to dataset YAML configuration file (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Validation batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device index or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on (default: val)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint '{args.model}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("RUNNING YOLO VALIDATION & EVALUATION")
    print(f"Model:       {model_path}")
    print(f"Data config: {args.data}")
    print(f"Split:       {args.split}")
    print(f"Image size:  {args.imgsz}")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Error: 'ultralytics' is not installed. Please install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or:\n"
            "  pip install ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 60)
    print(f"mAP@50:       {metrics.box.map50:.4f}")
    print(f"mAP@50-95:    {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
