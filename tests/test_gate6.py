"""Gate-6 Qwen alignment, LoRA drift, B4 attribution, Agent regression checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class Gate6Test(unittest.TestCase):
    def test_qwen_prompt_and_alignment(self) -> None:
        env = json.loads((RESULTS / "gate6_environment.json").read_text(encoding="utf-8"))
        self.assertEqual(env["qwen_revision"], "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
        self.assertEqual(env["prompt_sha256"], "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005")
        self.assertEqual(env["pooling"], "last_token")
        align = json.loads((RESULTS / "gate6_qwen_alignment.json").read_text(encoding="utf-8"))
        self.assertTrue(align["alignment_ok"])
        self.assertGreaterEqual(align["vector_alignment"]["mean_cosine"], 0.999)
        self.assertGreaterEqual(align["vector_alignment"]["min_cosine"], 0.995)
        self.assertEqual(align["vector_alignment"]["top1_intent_consistency"], 1.0)
        self.assertEqual(align["vector_alignment"]["direct_dim"], 1024)

    def test_qwen_lora_shapes_and_reload(self) -> None:
        plain = json.loads((RESULTS / "gate6_qwen_plain_training.json").read_text(encoding="utf-8"))
        anchor = json.loads((RESULTS / "gate6_qwen_anchor_training.json").read_text(encoding="utf-8"))
        self.assertEqual(plain["params"]["trainable"], 2293760)
        self.assertEqual(plain["epochs"], 2)
        self.assertEqual(plain["vector_check"]["dim"], 1024)
        self.assertLess(plain["vector_check"]["reload_max_abs"], 1e-6)
        self.assertGreater(anchor["vector_check"]["frozen_vs_lora_mean_cosine"], plain["vector_check"]["frozen_vs_lora_mean_cosine"])
        self.assertTrue((RESULTS / "gate6_qwen_plain_adapter" / "adapter_model.safetensors").exists())
        self.assertTrue((RESULTS / "gate6_qwen_anchor_adapter" / "adapter_model.safetensors").exists())

    def test_validation_gain_but_test_not_reproduced(self) -> None:
        val = pd.read_csv(RESULTS / "gate6_qwen_lora_validation.csv")
        frozen = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
        frozen = frozen[(frozen.dataset == "banking77") & (frozen.model == "qwen") & (frozen.shot == 10) & (frozen.method == "B3-H")].iloc[0]
        plain = val[(val.run == "plain") & (val.method == "B3-H")].iloc[0]
        self.assertGreater(float(plain["val_open_set_macro_f1"]), float(frozen["val_open_set_macro_f1"]))
        test = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_historical_test.csv")
        tf = test[(test.encoder == "qwen_frozen") & (test.method == "B3-H")].iloc[0]
        tp = test[(test.encoder == "qwen_plain_lora") & (test.method == "B3-H")].iloc[0]
        self.assertLess(float(tp["open_set_macro_f1"]), float(tf["open_set_macro_f1"]))
        self.assertTrue(test["historical_exploratory"].all())

    def test_qwen_b4_reject_beats_full(self) -> None:
        val = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_val.csv")
        for enc in ("qwen_frozen", "qwen_plain_lora"):
            r = val[(val.encoder == enc) & (val.method == "B4-RejectOnly")].iloc[0]
            f = val[(val.encoder == enc) & (val.method == "B4-Full")].iloc[0]
            self.assertGreater(float(r["open_set_macro_f1"]), float(f["open_set_macro_f1"]))
        payload = torch.load(RESULTS / "gate6_qwen_frozen_reject.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["meta"]["gate"], "fixed_0.5")
        self.assertEqual(payload["meta"]["encoder"], "frozen")

    def test_matrix_freeze_and_rows(self) -> None:
        val = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_val.csv")
        test = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_historical_test.csv")
        self.assertEqual(len(val), 9)
        self.assertEqual(len(test), 9)
        freeze = json.loads((RESULTS / "gate6_qwen_method_freeze.json").read_text(encoding="utf-8"))
        self.assertTrue(freeze["test_evaluated"])
        self.assertEqual(freeze["adapter_plain"], "4efaee5c891e26058764b4a7ec999dec718436f57b724841650831ddc01c3818")

    def test_agent_demos_are_qwen_lora(self) -> None:
        demos = json.loads((RESULTS / "gate6_qwen_agent_demos.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(demos), 4)
        lora = [d for d in demos if "Plain" in d["router"]]
        self.assertGreaterEqual(len(lora), 2)
        self.assertTrue(any(any(t["tools"] for t in d["turns"]) for d in demos))

    def test_qwen_agent_benchmark_four_routers(self) -> None:
        summary = pd.read_csv(RESULTS / "gate6_qwen_benchmark_summary.csv")
        results = pd.read_csv(RESULTS / "gate6_qwen_benchmark_results.csv")
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(results), 480)
        self.assertTrue((results["safe"] == 1).all())
        self.assertEqual(set(results["router"]), {"Qwen-Frozen-B3H", "Qwen-Plain-B3H", "Qwen-Plain-RejectOnly", "Qwen-Plain-B4Full"})
        reject = summary[summary.router == "Qwen-Plain-RejectOnly"].iloc[0]
        plain = summary[summary.router == "Qwen-Plain-B3H"].iloc[0]
        self.assertGreater(float(reject["task_success_rate"]), float(plain["task_success_rate"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
