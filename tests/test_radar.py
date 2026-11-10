"""Tests for Radar HUD mathematical transformations and scope rendering."""

import unittest
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.inference.radar import RadarHUD


class TestRadarHUD(unittest.TestCase):
    """Test Radar HUD calculations and coordinate mappings."""

    def test_missing_cv2_raises_import_error(self):
        if cv2 is None:
            with self.assertRaises(ImportError):
                RadarHUD(size=600)

    @unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed in current environment")
    def test_dimensions(self):
        hud = RadarHUD(size=600, ring_step=50)
        self.assertEqual(hud.size, 600)
        self.assertEqual(hud.center, (300, 300))
        self.assertEqual(hud.radius, 285)

    @unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed in current environment")
    def test_render_empty_scope(self):
        hud = RadarHUD(size=600, ring_step=50)
        scope = hud.render_scope([], frame_size=(1280, 720))
        self.assertEqual(scope.shape, (600, 600, 3))
        self.assertEqual(scope.dtype, np.uint8)

    @unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed in current environment")
    def test_target_plotting(self):
        hud = RadarHUD(size=600, ring_step=50)
        detections = [
            {"box": (600, 340, 680, 380), "label": "Drone", "conf": 0.88, "cls_id": 2},
            {"box": (100, 50, 200, 100), "label": "Aircrafts", "conf": 0.92, "cls_id": 0},
        ]
        scope = hud.render_scope(detections, frame_size=(1280, 720))
        self.assertEqual(scope.shape, (600, 600, 3))
        self.assertGreater(np.sum(scope), 0)

    @unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed in current environment")
    def test_combine_views(self):
        hud = RadarHUD(size=600, ring_step=50)
        camera_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        scope = hud.render_scope([], frame_size=(1280, 720))
        combined = hud.combine_views(camera_frame, scope)
        self.assertEqual(combined.shape, (600, 1200, 3))


if __name__ == "__main__":
    unittest.main()
