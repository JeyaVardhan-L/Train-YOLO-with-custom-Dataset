"""Airborne object detection (aircraft, birds, drones) with Ultralytics YOLO11.

Dataset preparation lives in the separate `yolotools` package.
"""

from .inference.detector import YOLOAirDetector
from .inference.radar import RadarHUD

__all__ = ["YOLOAirDetector", "RadarHUD"]
