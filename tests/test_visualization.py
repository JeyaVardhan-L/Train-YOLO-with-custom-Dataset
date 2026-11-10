"""Tests for visualization helpers and color mappings."""

import unittest
from src.utils.visualization import get_class_color, PALETTE


class TestVisualization(unittest.TestCase):
    """Test visualization palette consistency and indexing."""

    def test_palette_length(self):
        self.assertGreaterEqual(len(PALETTE), 3)

    def test_class_colors_deterministic(self):
        color0 = get_class_color(0)
        color1 = get_class_color(1)
        color2 = get_class_color(2)

        self.assertEqual(len(color0), 3)
        self.assertEqual(len(color1), 3)
        self.assertEqual(len(color2), 3)

        # Ensure distinct colors
        self.assertNotEqual(color0, color1)
        self.assertNotEqual(color1, color2)

    def test_color_modulo(self):
        # Out-of-range class ID should safely wrap around modulo length
        color_overflow = get_class_color(len(PALETTE) + 1)
        self.assertEqual(color_overflow, get_class_color(1))


if __name__ == "__main__":
    unittest.main()
