"""Modular YOLO detection engine for airborne and sky object detection."""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.utils.visualization import get_class_color


class YOLOAirDetector:
    """Wrapper around Ultralytics YOLO model tailored for airborne object detection."""

    def __init__(self, model_path: str = "weights/best.pt", device: Optional[str] = None):
        """Initialize detector with model weights.

        Args:
            model_path: Path to .pt model weights file.
            device: Computing device ('0', 'cpu', None for auto).
        """
        if YOLO is None:
            raise ImportError(
                "The 'ultralytics' library is required for object detection. "
                "Please install it using: pip install ultralytics"
            )
        if cv2 is None:
            raise ImportError(
                "The 'opencv-python' library is required for computer vision processing. "
                "Please install it using: pip install opencv-python"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Check canonical weights directory or standard base weights
            alt_paths = [
                Path("weights/best.pt"),
                Path("yolo11s.pt"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    print(f"Warning: Model not found at '{model_path}'. Using fallback '{alt}'.")
                    self.model_path = alt
                    break

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path), task="detect")
        self.names = self.model.names
        self.device = device

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Run object detection on an OpenCV BGR frame.

        Args:
            frame: BGR numpy image array.
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: Intersection over Union threshold for NMS.

        Returns:
            List of dicts: [{'box': (x1, y1, x2, y2), 'cls_id': int, 'label': str, 'conf': float}]
        """
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes = first_res.boxes
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()

        for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
            confidence = float(conf)
            if confidence >= conf_threshold:
                c_id = int(cls_id)
                x1, y1, x2, y2 = map(int, xyxy)
                if isinstance(self.names, dict):
                    label = self.names.get(c_id, f"Class_{c_id}")
                elif isinstance(self.names, (list, tuple)) and 0 <= c_id < len(self.names):
                    label = self.names[c_id]
                else:
                    label = f"Class_{c_id}"

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "cls_id": c_id,
                    "label": label,
                    "conf": confidence,
                })

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Draw bounding boxes, confidence tags, counts, and FPS onto the frame."""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det["cls_id"]
            label = det["label"]
            conf = det["conf"]
            color = get_class_color(cls_id)

            # Draw bounding rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label banner
            tag = f"{label} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            banner_y1 = max(0, y1 - th - baseline - 4)
            banner_y2 = y1
            banner_x2 = min(annotated.shape[1], x1 + tw + 6)

            cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 3, banner_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Header statistics
        hud_text = f"Detections: {len(detections)}"
        cv2.putText(annotated, hud_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if fps is not None and fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated
