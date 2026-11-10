import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Ultralytics YOLO model on custom airborne object detection dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data.yaml",
        help="Path to dataset YAML configuration file (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Initial model checkpoint or architecture (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
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
        help="CUDA device index (e.g., '0' or '0,1') or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader worker threads (default: 8)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save training run artifacts (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run experiment name (default: train)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint in the run directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset configuration file '{args.data}' not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("STARTING YOLO TRAINING")
    print(f"Data config:  {args.data}")
    print(f"Base model:   {args.model}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch}")
    print(f"Image size:   {args.imgsz}")
    print(f"Device:       {args.device if args.device else 'auto'}")
    print(f"Output:       {args.project}/{args.name}")
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

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        plots=True,
        save=True,
    )

    print("\nTraining completed successfully.")
    save_dir = Path(args.project) / args.name
    print(f"Results and checkpoints saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
