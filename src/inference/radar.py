"""Radar HUD visualization module for airborne object tracking."""

from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class RadarHUD:
    """Renders a circular tactical radar scope plotting detected airborne targets."""

    def __init__(self, size: int = 600, ring_step: int = 50):
        """Initialize radar display specifications.

        Args:
            size: Width and height of radar scope in pixels.
            ring_step: Distance in pixels between concentric range rings.
        """
        if cv2 is None:
            raise ImportError(
                "The 'opencv-python' library is required for Radar HUD visualization. "
                "Please install it using: pip install opencv-python"
            )

        self.size = size
        self.center = (size // 2, size // 2)
        self.radius = (size // 2) - 15
        self.ring_step = ring_step

    def render_scope(self, detections: List[Dict[str, Any]], frame_size: Tuple[int, int]) -> np.ndarray:
        """Render radar display frame and plot target blips based on frame detections.

        Args:
            detections: List of detection dictionaries containing 'box', 'label', 'conf'.
            frame_size: (width, height) of the optical camera frame.

        Returns:
            np.ndarray: BGR image of radar scope (size x size).
        """
        radar = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        frame_w, frame_h = frame_size
        cx, cy = self.center

        # 1. Dark grid background lines
        for x in range(0, self.size, 50):
            cv2.line(radar, (x, 0), (x, self.size), (0, 45, 0), 1)
        for y in range(0, self.size, 50):
            cv2.line(radar, (0, y), (self.size, y), (0, 45, 0), 1)

        # 2. Concentric range rings
        for r in range(self.ring_step, self.radius + 1, self.ring_step):
            cv2.circle(radar, self.center, r, (0, 110, 0), 1)

        # Outer boundary circle
        cv2.circle(radar, self.center, self.radius, (0, 200, 0), 2)

        # 3. Crosshairs
        cv2.line(radar, (cx, cy - self.radius), (cx, cy + self.radius), (0, 220, 0), 1)
        cv2.line(radar, (cx - self.radius, cy), (cx + self.radius, cy), (0, 220, 0), 1)

        # 4. Range ring labels
        cv2.putText(radar, "N", (cx - 5, cy - self.radius + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(radar, "S", (cx - 5, cy + self.radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(radar, "W", (cx - self.radius + 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(radar, "E", (cx + self.radius - 18, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # 5. Plot detection blips
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            obj_cx = (x1 + x2) / 2.0
            obj_cy = (y1 + y2) / 2.0

            # Normalized coordinate relative to camera center: [-1.0, 1.0]
            dx = (obj_cx - frame_w / 2.0) / (frame_w / 2.0) if frame_w > 0 else 0.0
            dy = (obj_cy - frame_h / 2.0) / (frame_h / 2.0) if frame_h > 0 else 0.0

            # Map to radar scope coordinates
            radar_x = int(cx + dx * (self.radius - 10))
            radar_y = int(cy + dy * (self.radius - 10))

            # Clamp within radar circle
            dist = np.sqrt((radar_x - cx) ** 2 + (radar_y - cy) ** 2)
            if dist > self.radius - 5:
                scale = (self.radius - 10) / dist
                radar_x = int(cx + (radar_x - cx) * scale)
                radar_y = int(cy + (radar_y - cy) * scale)

            # Draw target blip and strobe ring
            cv2.circle(radar, (radar_x, radar_y), 7, (0, 0, 255), -1)
            cv2.circle(radar, (radar_x, radar_y), 12, (0, 140, 255), 1)

            # Target label
            lbl = det.get("label", "Target")
            cv2.putText(
                radar,
                lbl,
                (radar_x + 10, radar_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Scope status header
        cv2.putText(radar, f"TACTICAL RADAR HUD | TRACKS: {len(detections)}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return radar

    def combine_views(self, optical_frame: np.ndarray, radar_scope: np.ndarray) -> np.ndarray:
        """Horizontally stack resized optical camera frame and radar scope side-by-side."""
        resized_optical = cv2.resize(optical_frame, (self.size, self.size))
        return np.hstack((resized_optical, radar_scope))
