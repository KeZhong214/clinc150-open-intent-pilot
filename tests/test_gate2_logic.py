"""Small model-free checks for Gate-2 threshold/decision code."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_gate2 import Scored, apply_method, metric_payload  # noqa: E402


class Gate2LogicTest(unittest.TestCase):
    def test_b2v2_accept_requires_both_conditions(self) -> None:
        classes = ["a", "b"]
        scored = Scored(
            sample_ids=np.array(["x"], dtype=object),
            roles=np.array(["final_known_test"], dtype=object),
            intents=np.array(["a"], dtype=object),
            s1=np.array([0.9], dtype=np.float32),
            s2=np.array([0.8], dtype=np.float32),
            margin=np.array([0.1], dtype=np.float32),
            pred_idx=np.array([0], dtype=int),
        )
        config = {"tau_c": {"a": 0.8, "b": 0.8}, "gamma": 0.2}
        pred, accept, decision = apply_method(scored, classes, "B2_v2", config)
        self.assertFalse(bool(accept[0]))
        self.assertEqual(pred[0], "OOS")
        self.assertAlmostEqual(float(decision[0]), -0.1, places=6)

    def test_metric_payload_allows_empty_original_oos(self) -> None:
        classes = ["a", "b"]
        empty = Scored(
            sample_ids=np.array([], dtype=object),
            roles=np.array([], dtype=object),
            intents=np.array([], dtype=object),
            s1=np.array([]),
            s2=np.array([]),
            margin=np.array([]),
            pred_idx=np.array([], dtype=int),
        )
        known = Scored(
            sample_ids=np.array(["k1", "k2"], dtype=object),
            roles=np.array(["final_known_test"] * 2, dtype=object),
            intents=np.array(["a", "b"], dtype=object),
            s1=np.array([0.9, 0.8], dtype=np.float32),
            s2=np.array([0.1, 0.2], dtype=np.float32),
            margin=np.array([0.8, 0.6], dtype=np.float32),
            pred_idx=np.array([0, 1], dtype=int),
        )
        hard = Scored(
            sample_ids=np.array(["h1", "h2"], dtype=object),
            roles=np.array(["hard_oos_test"] * 2, dtype=object),
            intents=np.array(["OOS", "OOS"], dtype=object),
            s1=np.array([0.2, 0.3], dtype=np.float32),
            s2=np.array([0.1, 0.2], dtype=np.float32),
            margin=np.array([0.1, 0.1], dtype=np.float32),
            pred_idx=np.array([0, 1], dtype=int),
        )
        payload = metric_payload(
            classes,
            known,
            np.array([0, 1]),
            empty,
            hard,
            np.array(["a", "b"], dtype=object),
            np.array([], dtype=object),
            np.array(["OOS", "OOS"], dtype=object),
            np.array([0.8, 0.6], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
        )
        self.assertIsNone(payload["orig_oos_recall"])
        self.assertEqual(payload["n_hard_oos"], 2)
        self.assertAlmostEqual(payload["hard_oos_recall"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
