"""Inference contracts using invented data, runnable in a clean checkout."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from demo import infer, synthetic_inputs


class IntentSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.query, self.audited, self.history = synthetic_inputs()

    def predict(self, **kwargs):
        return infer(self.query, self.audited, self.history, **kwargs)

    def test_variants_return_supported_joint_offset(self):
        for kwargs in ({}, {"outcome_weight": False}, {"shape_weight": False}):
            result = self.predict(**kwargs)
            self.assertEqual(result["status"], "provisional")
            self.assertGreaterEqual(result["effective_n"], 2.5)
            offset = [result["intent_x_01"] - self.query.model_target_x_01,
                      result["intent_y_01"] - self.query.model_target_y_01]
            observed = self.audited[["model_miss_x_01", "model_miss_y_01"]].to_numpy()
            self.assertTrue(np.any(np.all(np.isclose(observed, offset), axis=1)))
            self.assertAlmostEqual(result["intent_plate_x_ft"], (0.5-result["intent_x_01"])*17/12)
            self.assertAlmostEqual(result["intent_plate_z_ft"], 1.5+result["intent_y_01"]*2)
            self.assertAlmostEqual(sum(map(float, result["comparable_weights"].split(";"))), 1, places=5)

    def test_current_outcome_location_and_movement_are_unused(self):
        expected = self.predict()
        for column in ["plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed",
                       "release_spin_rate", "model_miss_x_01", "model_miss_y_01"]:
            setattr(self.query, column, 99999.0)
        self.query.description = "home_run"
        self.assertEqual(expected, self.predict())

    def test_same_date_future_and_unrelated_pitches_are_excluded(self):
        expected = self.predict()
        for changes in [dict(game_date=self.query.game_date), dict(game_date="2026-01-01"),
                        dict(pitcher=999), dict(pitch_type="SL")]:
            other_a = self.audited.assign(**changes, model_miss_x_01=999., model_miss_y_01=999.)
            other_h = self.history.assign(**changes, pfx_x=999., pfx_z=999.)
            self.assertEqual(expected, infer(self.query,
                pd.concat([self.audited, other_a]), pd.concat([self.history, other_h])))
        wrong_side = self.audited.assign(stand="L", model_miss_x_01=999.)
        result = infer(self.query, pd.concat([self.audited, wrong_side]), self.history)
        # The diagnostic group count includes both sides; eligible comparables do not.
        self.assertEqual(result.pop("prior_group_n"), 2 * len(self.audited))
        expected.pop("prior_group_n")
        self.assertEqual(expected, result)

    def test_insufficient_history_and_remote_targets_abstain(self):
        self.assertEqual(infer(self.query, self.audited, self.history.head(19))["status"], "insufficient_history")
        self.assertEqual(infer(self.query, self.audited.head(2), self.history)["status"], "insufficient_history")
        remote = SimpleNamespace(**{**vars(self.query), "model_target_x_01": 10.})
        self.assertEqual(infer(remote, self.audited, self.history)["status"], "insufficient_nearby_targets")

    def test_missing_profile_and_concentrated_support_abstain(self):
        self.assertEqual(infer(self.query, self.audited, self.history.assign(pfx_x=np.nan))["status"], "missing_shape_history")
        concentrated = self.audited.copy()
        concentrated.loc[concentrated.index[1:], "release_speed"] = 200.
        self.assertEqual(infer(self.query, concentrated, self.history)["status"], "insufficient_effective_support")

    def test_shuffling_rows_preserves_prediction(self):
        result = infer(self.query, self.audited.sample(frac=1, random_state=7),
                       self.history.sample(frac=1, random_state=7))
        for key in ["intent_x_01", "intent_y_01", "effective_n", "region_x_low_01", "region_y_high_01"]:
            self.assertAlmostEqual(self.predict()[key], result[key])


if __name__ == "__main__":
    unittest.main()
