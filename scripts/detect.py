#!/usr/bin/env python3
"""CLI Script to run YOLO airborne object detection on images, directories, videos, or webcams."""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Ensure project root is in sys.path when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import YOLOAirDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO airborne object detection on image, folder, video, or camera."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to trained model checkpoint (.pt) (default: weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image path ('test.jpg'), folder ('images/'), video ('video.mp4'), or camera index ('0' or 'usb0')",
    )
    parser.add_argument(
        "--conf",
        "--thresh",
        dest="conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Optional display/output resolution formatted as WxH (e.g., '1280x720')",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated detection results to disk",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/detect/predict",
        help="Directory to store saved results (default: runs/detect/predict)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display live OpenCV GUI window (useful for headless environments)",
    )
    return parser.parse_args()


def process_image(detector: YOLOAirDetector, image_path: Path, conf: float, iou: float, target_res, save_dir: Path, show: bool):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not read image '{image_path}'", file=sys.stderr)
        return

    if target_res:
        frame = cv2.resize(frame, target_res)

    t0 = time.perf_counter()
    detections = detector.detect_frame(frame, conf_threshold=conf, iou_threshold=iou)
    fps = 1.0 / max(1e-5, (time.perf_counter() - t0))

    annotated = detector.annotate_frame(frame, detections, fps=fps)
    print(f"[{image_path.name}] Found {len(detections)} airborne targets")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved result to: {out_path}")

    if show:
        cv2.imshow("YOLO Aerial Object Detection", annotated)
        cv2.waitKey(0)


def process_video_stream(detector: YOLOAirDetector, cap, conf: float, iou: float, target_res, save_path: Optional[Path], show: bool):
    writer = None
    fps_buffer = []

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_res = target_res if target_res else (w, h)
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_in, out_res)

    print("Starting video inference stream (press 'q' to stop, 's' to pause)...")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if target_res:
            frame = cv2.resize(frame, target_res)

        detections = detector.detect_frame(frame, conf_threshold=conf, iou_threshold=iou)

        fps_instant = 1.0 / max(1e-5, (time.perf_counter() - t0))
        fps_buffer.append(fps_instant)
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)

        annotated = detector.annotate_frame(frame, detections, fps=avg_fps)

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("YOLO Aerial Object Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.waitKey(0)

    if writer:
        writer.release()
        print(f"Video saved to: {save_path}")


def main():
    args = parse_args()
    if cv2 is None:
        print(
            "Error: 'opencv-python' is not installed. Please install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or:\n"
            "  pip install opencv-python",
            file=sys.stderr,
        )
        sys.exit(1)

    detector = YOLOAirDetector(model_path=args.model)

    target_res = None
    if args.resolution:
        try:
            rw, rh = map(int, args.resolution.lower().split("x"))
            target_res = (rw, rh)
        except Exception:
            print(f"Invalid resolution '{args.resolution}'. Expected WxH (e.g. 1280x720)", file=sys.stderr)
            sys.exit(1)

    src = args.source
    save_dir = Path(args.save_dir) if args.save else None
    show = not args.no_show

    # 1. Check if source is a camera index
    if src.isdigit() or src.startswith("usb"):
        cam_idx = int(src.replace("usb", ""))
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"Error: Unable to open camera device index {cam_idx}", file=sys.stderr)
            sys.exit(1)
        save_file = save_dir / "camera_recording.mp4" if save_dir else None
        try:
            process_video_stream(detector, cap, args.conf, args.iou, target_res, save_file, show)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    src_path = Path(src)
    if not src_path.exists():
        print(f"Error: Source '{src}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Check if source is a single image
    if src_path.is_file() and src_path.suffix in IMAGE_EXTS:
        process_image(detector, src_path, args.conf, args.iou, target_res, save_dir, show)
        cv2.destroyAllWindows()

    # 3. Check if source is a video file
    elif src_path.is_file() and src_path.suffix in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(src_path))
        save_file = save_dir / f"detected_{src_path.stem}.mp4" if save_dir else None
        try:
            process_video_stream(detector, cap, args.conf, args.iou, target_res, save_file, show)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # 4. Check if source is a directory of images
    elif src_path.is_dir():
        image_files = [p for p in src_path.iterdir() if p.suffix in IMAGE_EXTS]
        if not image_files:
            print(f"No valid image files found in '{src_path}'")
            return
        print(f"Processing {len(image_files)} images from folder '{src_path}'...")
        for img_p in image_files:
            process_image(detector, img_p, args.conf, args.iou, target_res, save_dir, show)
        cv2.destroyAllWindows()
    else:
        print(f"Unsupported source format: {src_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
