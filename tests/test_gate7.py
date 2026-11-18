"""Gate-7 development split, mechanism and multi-seed stability tests."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest"
RESULTS = ROOT / "results"

from src.gate4.agent import StubRouter  # noqa: E402
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402


class Gate7Test(unittest.TestCase):
    def test_dev_split_disjoint_and_frozen(self) -> None:
        thr = pd.read_csv(MANIFEST / "gate7_dev_threshold.csv")
        sel = pd.read_csv(MANIFEST / "gate7_dev_select.csv")
        info = json.loads((RESULTS / "gate7_dev_split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(thr), 760)
        self.assertEqual(len(sel), 760)
        self.assertEqual(int((thr.role == "calibration_known").sum()), 400)
        self.assertEqual(int((sel.role == "dev_unknown").sum()), 360)
        self.assertEqual(set(thr.sample_id).isdisjoint(set(sel.sample_id)), True)
        thr_classes = {x for x in thr[thr.role == "dev_unknown"].intent.unique()}
        sel_classes = {x for x in sel[sel.role == "dev_unknown"].intent.unique()}
        self.assertEqual(len(thr_classes), 9)
        self.assertEqual(len(sel_classes), 9)
        self.assertTrue(thr_classes.isdisjoint(sel_classes))
        for name in ("dev_threshold", "dev_select"):
            p = ROOT / info[name]["path"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), info[name]["sha256"])
        self.assertEqual(info["checks"]["sample_id_overlap"], 0)
        self.assertEqual(info["checks"]["text_sha256_overlap"], 0)
        self.assertNotIn("known_test", set(thr.role) | set(sel.role))
        self.assertNotIn("final_unknown", set(thr.role) | set(sel.role))

    def test_diagnosis_metrics_exist(self) -> None:
        d = pd.read_csv(RESULTS / "gate7_dev_diagnosis.csv")
        self.assertEqual(len(d), 8)
        plain = d[(d.encoder == "qwen_plain") & (d.method == "B3-H")].iloc[0]
        frozen = d[(d.encoder == "qwen_frozen") & (d.method == "B3-H")].iloc[0]
        self.assertGreater(float(plain.select_open_set_macro_f1), float(frozen.select_open_set_macro_f1))
        self.assertGreater(float(plain.select_oos_recall), float(frozen.select_oos_recall))

    def test_toy_batch_is_6x6_not_48x48(self) -> None:
        t = json.loads((RESULTS / "gate7_toy_batch.json").read_text(encoding="utf-8"))
        self.assertEqual(t["microbatch_similarity_shape"], [6, 6])
        self.assertEqual(t["grad_accum_8_hypothetical_shape"], [48, 48])
        self.assertEqual(t["leftover_microbatches"], 0)
        self.assertIn("never 48x48", t["actual_current_code"])

    def test_hard_negative_risk(self) -> None:
        h = json.loads((RESULTS / "gate7_hard_negative_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(h["n"], 400)
        self.assertGreater(h["margin_negative_count"], 200)
        self.assertGreater(h["duplicate_hard_negative_count"], 200)
        self.assertLess(h["margin_mean"], 0.0)

    def test_transfer_ambiguity_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = BankAgentV2(StubRouter("transfer_not_received_by_recipient", margin=0.5), session_id="g7-amb", db_dir=Path(tmp))
            try:
                turn = agent.handle("My transfer TXN-2001 has not arrived yet.")
                self.assertIn("not sure", turn.response.lower())
                self.assertEqual(turn.tools, [])
                self.assertTrue(any("transfer" in c for c in agent.state.clarification_candidates))
                turn2 = agent.handle("it has not arrived")
                self.assertTrue(turn2.tools)
                self.assertEqual(turn2.tools[0]["tool"], "get_transfer_status")
                self.assertIn("pending", turn2.response.lower())
                self.assertIn("TXN-2001", turn2.tools[0]["args"].get("transfer_id", ""))
            finally:
                agent.close()

    def test_final_test_frozen(self) -> None:
        freeze = json.loads((RESULTS / "gate7_final_freeze.json").read_text(encoding="utf-8"))
        self.assertTrue(freeze["test_evaluated"])
        test = pd.read_csv(RESULTS / "gate7_final_historical_test.csv")
        self.assertEqual(len(test), 12)
        self.assertTrue(test.historical_exploratory.all())
        summary = pd.read_csv(RESULTS / "gate7_final_summary.csv")
        self.assertEqual(len(summary), 8)
        reject = summary[(summary.method == "B4-RejectOnly") & (summary.metric == "open_set_macro_f1")].iloc[0]
        self.assertGreater(float(reject["delta_mean_vs_frozen"]), 0.0)
        self.assertGreaterEqual(int(reject["wins_vs_frozen"]), 2)

    def test_p3_completed_but_not_selected(self) -> None:
        p = json.loads((RESULTS / "gate7_p3_result.json").read_text(encoding="utf-8"))
        self.assertEqual(p["status"], "COMPLETED_NEGATIVE")
        self.assertFalse(p["selected"])
        self.assertEqual(p["optimizer_steps"], 50)
        self.assertTrue((RESULTS / "gate7_qwen_p3_adapter" / "adapter_model.safetensors").exists())
        self.assertLess(float(p["dev_b3h_f1"]), 0.7984)

    def test_stability_b4_reject_beats_full(self) -> None:
        d = pd.read_csv(RESULTS / "gate7_stability_summary.csv")
        for enc in ("gte_frozen", "qwen_frozen", "gte_lora42", "gte_lora43", "gte_lora44"):
            r = d[(d.encoder == enc) & (d.method == "B4-RejectOnly")]
            f = d[(d.encoder == enc) & (d.method == "B4-Full")]
            if len(r) and len(f):
                self.assertGreater(float(r.dev_open_set_macro_f1_mean.iloc[0]), float(f.dev_open_set_macro_f1_mean.iloc[0]))
        hashes = pd.read_csv(RESULTS / "gate7_stability_checkpoint_hashes.csv")
        self.assertGreaterEqual(len(hashes), 30)
        self.assertEqual(hashes.sha256.nunique(), len(hashes))

    def test_agent_three_router_regression(self) -> None:
        s = pd.read_csv(RESULTS / "gate7_agent_summary.csv")
        self.assertEqual(len(s), 3)
        d = pd.read_csv(RESULTS / "gate7_agent_results.csv")
        self.assertEqual(len(d), 360)
        self.assertTrue((d.safe == 1).all())
        qwen = s[s.router == "Qwen-Frozen-RejectOnly"].iloc[0]
        gte = s[s.router == "GTE-Frozen-RejectOnly"].iloc[0]
        self.assertGreater(float(qwen.task_success_rate), float(gte.task_success_rate))
        self.assertEqual(float(qwen.clarification_success), 1.0)

    def test_protocol_fix_thresholds(self) -> None:
        freeze = json.loads((RESULTS / "gate7_final_freeze_v2.json").read_text(encoding="utf-8"))
        self.assertTrue(freeze["protocol_fix_v2"])
        self.assertTrue(freeze["test_evaluated"])
        self.assertEqual(len(freeze["thresholds"]), 12)
        for row in freeze["thresholds"]:
            dev = pd.read_csv(RESULTS / f"gate7_dev_{row['encoder']}.csv")
            expected = float(dev[dev.method == row["method"]].tau.iloc[0])
            self.assertAlmostEqual(float(row["tau_used"]), expected, places=12)
            self.assertAlmostEqual(float(row["dev_threshold_tau"]), expected, places=12)
        old = json.loads((RESULTS / "gate7_final_freeze_pre_fix.json").read_text(encoding="utf-8"))
        self.assertTrue(old["test_evaluated"])
        self.assertTrue((RESULTS / "gate7_final_historical_test_pre_fix.csv").exists())
        manifest = json.loads((RESULTS / "gate7_protocol_fix_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"]["results/gate7_final_historical_test_pre_fix.csv"], hashlib.sha256((RESULTS / "gate7_final_historical_test_pre_fix.csv").read_bytes()).hexdigest())
        old_test = pd.read_csv(RESULTS / "gate7_final_historical_test_pre_fix.csv")
        new_test = pd.read_csv(RESULTS / "gate7_final_historical_test_v2.csv")
        for method in ("B0", "B3-H", "B4-RejectOnly"):
            a = old_test[old_test.method == method].sort_values("encoder").tau.to_numpy()
            b = new_test[new_test.method == method].sort_values("encoder").tau.to_numpy()
            if method != "B4-RejectOnly":
                self.assertTrue((a == b).all())
            else:
                self.assertFalse((a == b).all())

    def test_stability_36_unique(self) -> None:
        d = pd.read_csv(RESULTS / "gate7_stability_b4_results.csv")
        self.assertEqual(len(d), 36)
        self.assertEqual(d.duplicated(["encoder", "method", "seed"]).sum(), 0)
        full = d[(d.encoder == "qwen_plain42") & (d.method == "B4-Full")]
        self.assertEqual(sorted(full.seed.tolist()), [42, 43, 44])
        ckpt = RESULTS / "gate7_stability_qwen_plain42_full_seed44.pt"
        self.assertTrue(ckpt.exists())
        hashes = pd.read_csv(RESULTS / "gate7_stability_checkpoint_hashes.csv")
        self.assertEqual(hashes.sha256.nunique(), len(hashes))
        supp = json.loads((RESULTS / "gate7_stability_supplement.json").read_text(encoding="utf-8"))
        self.assertEqual(supp["pre_supplement_rows"], 35)
        self.assertEqual(supp["final_rows"], 36)
        self.assertEqual(supp["pre_supplement_sha256"], hashlib.sha256((RESULTS / "gate7_stability_b4_results_pre_supplement.csv").read_bytes()).hexdigest())

    def test_agent_accurate_name_and_threshold_source(self) -> None:
        s = pd.read_csv(RESULTS / "gate7_agent_summary.csv")
        row = s[s.router == "Qwen-Plain42-RejectOnly"].iloc[0]
        self.assertEqual(row.router_legacy, "Qwen-Opt-RejectOnly")
        config = (ROOT / "configs" / "gate7_agent.yaml").read_text(encoding="utf-8")
        self.assertIn("Qwen-Plain42-RejectOnly", config)
        self.assertIn("Qwen-Opt-RejectOnly", config)
        # The Agent run still uses Gate5/6 thresholds; this is explicitly a
        # separate protocol from the Gate7 classification test.
        self.assertTrue((RESULTS / "gate5_reject_only_frozen_threshold.json").exists())
        self.assertTrue((RESULTS / "gate6_qwen_frozen_reject_threshold.json").exists())

    def test_summaries_match_raw(self) -> None:
        v2 = pd.read_csv(RESULTS / "gate7_final_historical_test_v2.csv")
        summary = pd.read_csv(RESULTS / "gate7_final_summary.csv")
        for method in ("B3-H", "B4-RejectOnly"):
            raw = v2[(v2.method == method) & v2.encoder.str.startswith("plain")].open_set_macro_f1.mean()
            got = summary[(summary.method == method) & (summary.metric == "open_set_macro_f1")].iloc[0]["mean"]
            self.assertAlmostEqual(float(raw), float(got), places=10)
        st = pd.read_csv(RESULTS / "gate7_stability_b4_results.csv")
        ss = pd.read_csv(RESULTS / "gate7_stability_summary.csv")
        raw = st[(st.encoder == "qwen_plain42") & (st.method == "B4-RejectOnly")].dev_open_set_macro_f1.mean()
        got = ss[(ss.encoder == "qwen_plain42") & (ss.method == "B4-RejectOnly")].dev_open_set_macro_f1_mean.iloc[0]
        self.assertAlmostEqual(float(raw), float(got), places=10)

    def test_all_historical_results_exploratory(self) -> None:
        for name in ("gate7_final_historical_test.csv", "gate7_final_historical_test_v2.csv", "gate7_stability_historical_test.csv"):
            d = pd.read_csv(RESULTS / name)
            self.assertTrue(d.historical_exploratory.all(), name)
        # Filled after B; ensures each trained seed has an independent adapter.
        if (RESULTS / "gate7_seed_runs.csv").exists():
            d = pd.read_csv(RESULTS / "gate7_seed_runs.csv")
            self.assertTrue({"seed", "tag", "adapter_sha256", "epochs"}.issubset(d.columns))
            self.assertGreaterEqual(d.seed.nunique(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
