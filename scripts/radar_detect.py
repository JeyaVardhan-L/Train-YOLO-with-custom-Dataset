#!/usr/bin/env python3
"""Tactical Radar HUD detection display for airborne targets."""

import argparse
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import YOLOAirDetector
from src.inference.radar import RadarHUD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live YOLO airborne detection with tactical 2D Radar HUD view."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to trained YOLO weights file (default: weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index ('0', '1', 'usb0') or video file path (default: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Detection confidence threshold (default: 0.45)",
    )
    parser.add_argument(
        "--radar-size",
        type=int,
        default=600,
        help="Radar scope resolution in pixels (default: 600)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to record combined video output (e.g., 'radar_output.mp4')",
    )
    return parser.parse_args()


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

    print("=" * 60)
    print("AIRBORNE TARGET TRACKING - RADAR HUD INTERFACE")
    print(f"Model:      {args.model}")
    print(f"Source:     {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"Scope size: {args.radar_size}x{args.radar_size}")
    print("=" * 60)

    detector = YOLOAirDetector(model_path=args.model)
    radar = RadarHUD(size=args.radar_size)

    # Resolve capture source
    if args.source.isdigit() or args.source.startswith("usb"):
        cam_idx = int(args.source.replace("usb", ""))
        cap = cv2.VideoCapture(cam_idx)
    else:
        src_p = Path(args.source)
        if not src_p.exists():
            print(f"Error: Source file '{args.source}' does not exist.", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(str(src_p))

    if not cap.isOpened():
        print(f"Error: Could not connect to video capture device / file '{args.source}'.", file=sys.stderr)
        print("Tip: If using webcam, try --source 0 or --source 1.", file=sys.stderr)
        sys.exit(1)

    writer = None
    if args.save:
        save_p = Path(args.save)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Combined view is 2 * radar_size wide, radar_size high
        writer = cv2.VideoWriter(str(save_p), fourcc, fps, (args.radar_size * 2, args.radar_size))

    print("Live radar feed running. Press 'q' to quit.")

    fps_buffer = []
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of stream or no frame received.")
                break

            h, w = frame.shape[:2]
            detections = detector.detect_frame(frame, conf_threshold=args.conf)

            fps_instant = 1.0 / max(1e-5, (time.perf_counter() - t0))
            fps_buffer.append(fps_instant)
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer)

            # 1. Annotate optical frame
            annotated_cam = detector.annotate_frame(frame, detections, fps=avg_fps)

            # 2. Render radar HUD scope
            radar_scope = radar.render_scope(detections, frame_size=(w, h))

            # 3. Stack views horizontally
            combined = radar.combine_views(annotated_cam, radar_scope)

            if writer:
                writer.write(combined)

            cv2.imshow("YOLO Aerial Detection + Tactical Radar HUD", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Radar video saved to: {args.save}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
