import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refine_broadcast_zone import KEYS, ZoneRefiner, box_edges, finite_box, refine_zone, register_neighbor
from apply_calibrated_zones_to_predictions import recompute_target, validate_zone_frames
from build_statcast_zone_predictions import choose_glove_row


class RectangleFitTests(unittest.TestCase):
    def setUp(self):
        self.truth = dict(zone_x=640, zone_y=300, zone_width=64, zone_height=88)
        self.prior = dict(zone_x=649, zone_y=293, zone_width=72, zone_height=99)
        rng = np.random.default_rng(12)
        self.image = rng.integers(20, 65, (720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (608, 256), (672, 344), (190, 190, 190), 1)

    def test_corrects_all_four_edges(self):
        result = refine_zone(self.image, self.prior)
        self.assertTrue(result['accepted'])
        np.testing.assert_allclose(box_edges(result), box_edges(self.truth), atol=1)

    def test_corrects_scaled_resolution(self):
        image = cv2.resize(self.image, (2560, 1440))
        result = refine_zone(image, {k: v*2 for k,v in self.prior.items()})
        self.assertTrue(result['accepted'])
        np.testing.assert_allclose(box_edges(result), box_edges(self.truth)*2, atol=2)

    def test_rejects_missing_rectangle(self):
        image = np.full_like(self.image, 40)
        result = refine_zone(image, self.prior)
        self.assertFalse(result['accepted'])

    def test_rejects_only_vertical_lines(self):
        image = np.full_like(self.image, 40)
        for x in (608, 672):
            cv2.line(image, (x,256), (x,344), (190,190,190), 1)
        self.assertFalse(refine_zone(image, self.prior)['accepted'])

    def test_rejects_solid_white_uniform(self):
        image = np.full_like(self.image, 40)
        cv2.rectangle(image, (608,256), (672,344), (230,230,230), -1)
        self.assertFalse(refine_zone(image, self.prior)['accepted'])

    def test_invalid_geometry_and_missing_image(self):
        for value in [float('nan'), float('inf'), -1, 0]:
            self.assertFalse(finite_box({**self.truth, 'zone_width': value}, self.image.shape))
        self.assertIsNone(refine_zone(None, self.prior))

    def test_missing_image_stays_unverified(self):
        dense = pd.DataFrame(columns=['pitch_uid','frame_uid'])
        result, evidence = ZoneRefiner(dense, '/tmp').resolve('a','a_1','nonexistent_zone_test.jpg',self.truth)
        self.assertEqual(result, self.truth)
        self.assertEqual(evidence['zone_refinement_status'], 'missing_image')

    def test_selected_roboflow_precedes_wrong_game_template(self):
        with tempfile.TemporaryDirectory() as directory:
            cv2.imwrite(str(Path(directory)/'frame.png'), self.image)
            row = dict(pitch_uid='p',frame_uid='p_1',image_path='frame.png',**self.prior)
            resolver = ZoneRefiner(pd.DataFrame([row]), directory)
            bad_game = dict(zone_x=710, zone_y=390, zone_width=90, zone_height=110)
            result, evidence = resolver.resolve('p','p_1','frame.png',bad_game)
            np.testing.assert_allclose(box_edges(result), box_edges(self.truth), atol=1)
            self.assertEqual(evidence['zone_baseline_source'], 'roboflow_selected_frame')

    def test_camera_translation_registration(self):
        moved = cv2.warpAffine(self.image, np.float32([[1,0,4],[0,1,-3]]), (1280,720))
        result = register_neighbor(self.image, moved, self.truth)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(box_edges(result), box_edges(self.truth)+[4,-3,4,-3], atol=.5)

    def test_camera_cut_rejected(self):
        self.assertIsNone(register_neighbor(self.image, np.zeros_like(self.image), self.truth))

    def test_wrong_frame_merge_rejected(self):
        predictions = pd.DataFrame([dict(pitch_uid='p', frame_uid='p_1')])
        zones = pd.DataFrame([dict(pitch_uid='p', frame_uid='p_2')])
        with self.assertRaises(ValueError):
            validate_zone_frames(predictions, zones)

    def test_duplicate_pitch_rejected(self):
        rows = pd.DataFrame([dict(pitch_uid='p'), dict(pitch_uid='p')])
        with self.assertRaises(ValueError):
            validate_zone_frames(rows, rows)

    def test_rebuilding_keeps_selected_frame_despite_low_glove_confidence(self):
        primary = pd.Series(dict(frame_uid='p_1', glove_confidence=.1, glove_x=630, glove_y=300))
        fallback = pd.Series(dict(frame_uid='p_2', glove_confidence=.9, glove_x=650, glove_y=330))
        selected, _ = choose_glove_row(primary, fallback, .5, .1)
        self.assertEqual(selected.frame_uid, 'p_1')

    def test_target_recomputed_from_corrected_geometry(self):
        row = pd.Series({**self.truth, 'glove_x':608,'glove_y':256})
        self.assertEqual(recompute_target(row), (0,1))
        row['zone_width']=0
        self.assertEqual(recompute_target(row), (None,None))


if __name__ == '__main__':
    unittest.main()
