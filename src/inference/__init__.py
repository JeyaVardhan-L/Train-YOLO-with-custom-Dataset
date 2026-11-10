"""Inference modules for standard detection and radar HUD visualization."""

from .detector import YOLOAirDetector
from .radar import RadarHUD

__all__ = ["YOLOAirDetector", "RadarHUD"]
