import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fox_zone_crop import crop_zone, restore_zone


class FoxZoneCropTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_native_crop_and_coordinate_roundtrip(self):
        self.image[284, 617] = [20, 40, 60]
        crop, origin = crop_zone(self.image, dict(zone_x='650', zone_y='320'))
        self.assertEqual(crop.shape, (320, 320, 3))
        self.assertEqual(origin, dict(left=490, top=160, width=320, height=320))
        np.testing.assert_array_equal(crop[124, 127], [20, 40, 60])
        result = restore_zone(dict(x=157.5, y=165.5, width=61, height=83), origin)
        self.assertEqual(result, dict(zone_x=647.5, zone_y=325.5,
                                      zone_width=61, zone_height=83))

    def test_invalid_priors_use_safe_center(self):
        for prior in ({}, dict(zone_x='bad', zone_y=None),
                      dict(zone_x=float('nan'), zone_y=float('inf')),
                      dict(zone_x=1000, zone_y=600)):
            _, origin = crop_zone(self.image, prior)
            self.assertEqual((origin['left'], origin['top']), (480, 140))

    def test_wrong_image_resolution_rejected(self):
        for image in (None, np.zeros((360, 640, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                crop_zone(image, {})

    def test_invalid_or_truncated_prediction_rejected(self):
        _, origin = crop_zone(self.image, {})
        good = dict(x=160, y=160, width=60, height=80)
        for prediction in ({}, {**good, 'x': float('nan')},
                           {**good, 'width': -1}, {**good, 'x': 10},
                           {**good, 'y': 310}, {**good, 'height': 'bad'}):
            self.assertIsNone(restore_zone(prediction, origin))


if __name__ == '__main__':
    unittest.main()
