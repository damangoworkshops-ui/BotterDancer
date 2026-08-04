"""Workload-Erkennung + Baseline-Skalierung des Dashboards (Audit 04.08. F8)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import dashboard  # noqa: E402

FINAL_GRAPH = {
    "1": {"class_type": "WanAnimateToVideo",
          "inputs": {"width": 512, "height": 768, "length": 81}},
    "2": {"class_type": "KSampler", "inputs": {"steps": 20}},
}
DRAFT_GRAPH = {
    "1": {"class_type": "WanAnimateToVideo",
          "inputs": {"width": 512, "height": 768, "length": 81}},
    "2": {"class_type": "KSampler", "inputs": {"steps": 4}},
    "3": {"class_type": "LoraLoaderModelOnly",
          "inputs": {"lora_name": "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"}},
}


class TestWorkload(unittest.TestCase):
    def test_final(self):
        wl = dashboard.workload_from_prompt(FINAL_GRAPH)
        self.assertEqual((wl["width"], wl["height"], wl["length"]), (512, 768, 81))
        self.assertFalse(wl["draft"])

    def test_draft_erkannt(self):
        self.assertTrue(dashboard.workload_from_prompt(DRAFT_GRAPH)["draft"])

    def test_unbrauchbarer_graph(self):
        self.assertIsNone(dashboard.workload_from_prompt(None))
        self.assertIsNone(dashboard.workload_from_prompt({"1": {"class_type": "KSampler",
                                                                "inputs": {"steps": 20}}}))


class TestBaseline(unittest.TestCase):
    def test_referenz_workload(self):
        wl = dashboard.workload_from_prompt(FINAL_GRAPH)
        self.assertAlmostEqual(dashboard.baseline_for(wl), 15.7, places=2)

    def test_draft_niedrigere_basis(self):
        wl = dashboard.workload_from_prompt(DRAFT_GRAPH)
        self.assertAlmostEqual(dashboard.baseline_for(wl), 8.0, places=2)

    def test_skalierung_mit_workload(self):
        wl = dict(dashboard.workload_from_prompt(FINAL_GRAPH))
        wl["length"] = 162  # doppelte Framezahl -> doppelte s/it-Erwartung
        self.assertAlmostEqual(dashboard.baseline_for(wl), 31.4, places=2)

    def test_fallback_ohne_workload(self):
        self.assertEqual(dashboard.baseline_for(None), dashboard.BASELINE_S_IT)


if __name__ == "__main__":
    unittest.main()
