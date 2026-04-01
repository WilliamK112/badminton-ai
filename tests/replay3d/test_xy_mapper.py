from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.replay3d.schema import CourtSpec
from src.replay3d.xy_mapper import build_homography_from_corners, map_image_point_to_court


class TestReplay3DXYMapper(unittest.TestCase):
    def test_maps_corners_and_center_in_meter_units(self) -> None:
        court = CourtSpec(length_m=13.4, width_m=6.1)

        # Synthetic top-down-like rectangle in image pixels.
        # 100 px ~= 1 meter scale in both axes.
        image_corners_xy = [
            [100.0, 200.0],   # bottom_left  -> (0.0, 0.0)
            [710.0, 200.0],   # bottom_right -> (6.1, 0.0)
            [710.0, 1540.0],  # top_right    -> (6.1, 13.4)
            [100.0, 1540.0],  # top_left     -> (0.0, 13.4)
        ]

        homography = build_homography_from_corners(image_corners_xy, court)

        bl = map_image_point_to_court((100.0, 200.0), homography)
        tr = map_image_point_to_court((710.0, 1540.0), homography)
        center = map_image_point_to_court((405.0, 870.0), homography)

        self.assertAlmostEqual(bl[0], 0.0, places=4)
        self.assertAlmostEqual(bl[1], 0.0, places=4)

        self.assertAlmostEqual(tr[0], 6.1, places=4)
        self.assertAlmostEqual(tr[1], 13.4, places=4)

        # Center point should map to half court width/length in meters.
        self.assertAlmostEqual(center[0], 3.05, places=3)
        self.assertAlmostEqual(center[1], 6.70, places=3)


if __name__ == "__main__":
    unittest.main()
