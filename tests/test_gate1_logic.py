"""Small, model-free checks for Gate-1 threshold and metric logic."""

from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_baseline import Scores, metrics, predict, select_b0  # noqa: E402


class Gate1LogicTest(unittest.TestCase):
    def test_b0_threshold_respects_acceptance_constraint(self) -> None:
        known = np.array([0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72])
        oos = np.array([0.87, 0.81, 0.75, 0.69])
        tau, info, grid = select_b0(known, oos)
        self.assertTrue(info["constraint_met"])
        self.assertGreaterEqual(info["known_acceptance_val"], 0.90 - 1e-12)
        self.assertAlmostEqual(info["known_acceptance_val"], float(np.mean(known >= tau)))
        self.assertEqual(len(grid), 101)

    def test_metrics_on_toy_predictions(self) -> None:
        known_classes = ["a", "b"]
        # 4 known + 2 OOS; B0 accepts three known and rejects one OOS.
        sample_id = np.array(["k1", "k2", "k3", "k4", "o1", "o2"], dtype=object)
        role = np.array(
            ["final_known_test", "final_known_test", "final_known_test", "final_known_test",
             "final_oos_test", "final_oos_test"],
            dtype=object,
        )
        label = np.array(["a", "b", "a", "b", "OOS", "OOS"], dtype=object)
        pred_idx = np.array([0, 1, 0, 1, 0, 0], dtype=int)
        scores = Scores(
            sample_id=sample_id,
            role=role,
            label=label,
            vectors=np.zeros((6, 2), dtype=np.float32),
            s1=np.array([0.9, 0.8, 0.7, 0.6, 0.72, 0.55], dtype=np.float32),
            s2=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5], dtype=np.float32),
            margin=np.array([0.8, 0.6, 0.4, 0.2, 0.45, 0.25], dtype=np.float32),
            pred_idx=pred_idx,
        )
        prediction = predict(
            scores,
            known_classes,
            method="B0",
            tau=0.65,
            q_c=np.zeros(2),
            lam=0.0,
            gamma=0.0,
        )
        row, cm, per_class = metrics(
            scores,
            known_classes,
            prediction,
            np.ones(6, dtype=bool),
        )
        self.assertEqual(row["n_known"], 4)
        self.assertEqual(row["n_oos"], 2)
        self.assertAlmostEqual(row["known_acceptance"], 0.75)
        self.assertAlmostEqual(row["oos_recall"], 0.5)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual([item["intent"] for item in per_class], ["a", "b"])

    def test_b2_prediction_requires_both_conditions(self) -> None:
        known_classes = ["a", "b"]
        sample_id = np.array(["x"], dtype=object)
        role = np.array(["final_known_test"], dtype=object)
        label = np.array(["a"], dtype=object)
        scores = Scores(
            sample_id=sample_id,
            role=role,
            label=label,
            vectors=np.zeros((1, 2), dtype=np.float32),
            s1=np.array([0.9], dtype=np.float32),
            s2=np.array([0.8], dtype=np.float32),
            margin=np.array([0.1], dtype=np.float32),
            pred_idx=np.array([0], dtype=int),
        )
        prediction = predict(
            scores,
            known_classes,
            method="B2",
            tau=0.5,
            q_c=np.array([0.7, 0.7]),
            lam=0.0,
            gamma=0.2,
        )
        self.assertFalse(bool(prediction["accept"][0]))
        self.assertEqual(prediction["pred_labels"][0], "OOS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
