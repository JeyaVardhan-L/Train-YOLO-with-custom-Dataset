"""Visualization utilities for bounding boxes, overlays, and color palettes."""

from typing import Tuple, List, Dict

# Distinct, high-contrast BGR colors for detection display
PALETTE: List[Tuple[int, int, int]] = [
    (0, 165, 255),   # Aircrafts: Orange (BGR)
    (255, 191, 0),   # Bird: Deep Sky Blue / Cyan (BGR)
    (50, 205, 50),   # Drone: Lime Green (BGR)
    (147, 20, 255),  # Extra class fallback: Pink
    (0, 215, 255),   # Extra class fallback: Gold
]

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR color tuple for a given class ID."""
    return PALETTE[class_id % len(PALETTE)]
