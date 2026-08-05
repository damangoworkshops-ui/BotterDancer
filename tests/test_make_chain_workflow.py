"""Vertragstests fuer den Chain-Workflow-Generator (lange Routinen als Chunks).

Nachgereicht 06.08. — das Tool ging entgegen der eigenen Regel ohne Tests in
den Commit. Testet den generierten Graphen strukturell (kein GPU/Server noetig).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(PROJECT, "tools", "make_chain_workflow.py")

BASE_GRAPH = {
    "prompt": {
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "x.safetensors"}},
        "11": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["10", 0], "shift": 8.0}},
        "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg"}},
        "15": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
        "17": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
        "20": {"class_type": "VHS_LoadImagesPath",
               "inputs": {"directory": "alt", "image_load_cap": 81,
                          "skip_first_images": 0, "select_every_nth": 1}},
        "30": {"class_type": "WanAnimateToVideo",
               "inputs": {"positive": ["13", 0], "negative": ["14", 0], "vae": ["15", 0],
                          "width": 512, "height": 768, "length": 81, "batch_size": 1,
                          "continue_motion_max_frames": 5, "video_frame_offset": 0,
                          "reference_image": ["17", 0], "pose_video": ["20", 0]}},
        "31": {"class_type": "KSampler",
               "inputs": {"model": ["11", 0], "positive": ["30", 0], "negative": ["30", 1],
                          "latent_image": ["30", 2], "seed": 42, "steps": 20}},
        "35": {"class_type": "TrimVideoLatent",
               "inputs": {"samples": ["31", 0], "trim_amount": ["30", 3]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["15", 0]}},
        "33": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["32", 0], "frame_rate": 16,
                          "filename_prefix": "alt", "save_metadata": False}},
    }
}


def run_tool(base_path, out_path, chunks, extra=()):
    return subprocess.run(
        [sys.executable, TOOL, "--base", base_path, "--chunks", str(chunks),
         "--pose-dir", r"C:\pose", "--pose-frames", "160", "--prefix", "chain_test",
         "--out", out_path, *extra],
        capture_output=True, text=True, timeout=60)


class TestChainGenerator(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.base = os.path.join(self.d, "base.json")
        self.out = os.path.join(self.d, "chain.json")
        with open(self.base, "w") as f:
            json.dump(BASE_GRAPH, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.d, ignore_errors=True)

    def _gen(self, chunks, extra=()):
        r = run_tool(self.base, self.out, chunks, extra)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out) as f:
            return json.load(f)["prompt"]

    def _nodes(self, p, ct):
        return [k for k, n in p.items() if n.get("class_type") == ct]

    def test_chunk_anzahl_und_parameter(self):
        p = self._gen(3)
        self.assertEqual(len(self._nodes(p, "WanAnimateToVideo")), 3)
        self.assertEqual(len(self._nodes(p, "KSampler")), 3)
        self.assertEqual(len(self._nodes(p, "VAEDecode")), 3)
        # geteilte Ressourcen bleiben einmalig (Review-Fix #5)
        self.assertEqual(len(self._nodes(p, "LoadImage")), 1)
        self.assertEqual(len(self._nodes(p, "VHS_LoadImagesPath")), 1)
        self.assertEqual(len(self._nodes(p, "VHS_VideoCombine")), 1)
        self.assertEqual(p["20"]["inputs"]["directory"], r"C:\pose")
        self.assertEqual(p["20"]["inputs"]["image_load_cap"], 160)
        self.assertEqual(p["33"]["inputs"]["filename_prefix"], "chain_test")

    def test_chain_verkabelung(self):
        """Kernvertrag: Chunk N+1 zieht video_frame_offset aus Output 5 des
        Vorgaengers und continue_motion aus dessen VAEDecode."""
        p = self._gen(3)
        wan = sorted(self._nodes(p, "WanAnimateToVideo"), key=lambda k: (len(k), k))
        self.assertEqual(p[wan[0]]["inputs"]["video_frame_offset"], 0)  # erster Chunk
        self.assertNotIn("continue_motion", p[wan[0]]["inputs"])
        for prev, cur in zip(wan, wan[1:]):
            off = p[cur]["inputs"]["video_frame_offset"]
            self.assertEqual(off, [prev, 5], f"{cur}: {off}")
            cm = p[cur]["inputs"]["continue_motion"]
            self.assertEqual(p[cm[0]]["class_type"], "VAEDecode")
            # der Decode muss zum VORGAENGER-Chunk gehoeren
            trim = p[cm[0]]["inputs"]["samples"][0]
            self.assertEqual(p[trim]["inputs"]["trim_amount"][0], prev)

    def test_interne_verkabelung_pro_chunk(self):
        """Jede Kopie zeigt auf ihre EIGENEN Nachbarn, nicht auf Chunk 1."""
        p = self._gen(2)
        wan2 = [k for k in self._nodes(p, "WanAnimateToVideo") if k != "30"][0]
        ks2 = [k for k in self._nodes(p, "KSampler") if k != "31"][0]
        trim2 = [k for k in self._nodes(p, "TrimVideoLatent") if k != "35"][0]
        dec2 = [k for k in self._nodes(p, "VAEDecode") if k != "32"][0]
        self.assertEqual(p[ks2]["inputs"]["latent_image"], [wan2, 2])
        self.assertEqual(p[trim2]["inputs"]["samples"], [ks2, 0])
        self.assertEqual(p[trim2]["inputs"]["trim_amount"], [wan2, 3])
        self.assertEqual(p[dec2]["inputs"]["samples"], [trim2, 0])
        # geteilte Loader bleiben referenziert
        self.assertEqual(p[ks2]["inputs"]["model"], ["11", 0])
        self.assertEqual(p[wan2]["inputs"]["reference_image"], ["17", 0])

    def test_imagebatch_kaskade_reihenfolge(self):
        """Alle Chunk-Bilder landen in EINEM VideoCombine, chronologisch."""
        p = self._gen(3)
        src = p["33"]["inputs"]["images"]
        order = []
        while p[src[0]]["class_type"] == "ImageBatch":
            node = p[src[0]]
            order.insert(0, node["inputs"]["image2"][0])
            src = node["inputs"]["image1"]
        order.insert(0, src[0])
        self.assertEqual(len(order), 3)
        self.assertEqual(order[0], "32")  # Chunk 1 zuerst
        self.assertEqual(len(set(order)), 3)
        for d in order:
            self.assertEqual(p[d]["class_type"], "VAEDecode")

    def test_ein_chunk_ist_basisgraph(self):
        p = self._gen(1)
        self.assertEqual(len(self._nodes(p, "WanAnimateToVideo")), 1)
        self.assertEqual(len(self._nodes(p, "ImageBatch")), 0)
        self.assertEqual(p["33"]["inputs"]["images"], ["32", 0])

    def test_chunk_length_wird_gesetzt(self):
        p = self._gen(2, extra=("--chunk-length", "77"))
        for w in self._nodes(p, "WanAnimateToVideo"):
            self.assertEqual(p[w]["inputs"]["length"], 77)

    def test_mehrdeutiger_basisgraph_wird_abgelehnt(self):
        doubled = json.loads(json.dumps(BASE_GRAPH))
        doubled["prompt"]["99"] = json.loads(json.dumps(BASE_GRAPH["prompt"]["30"]))
        with open(self.base, "w") as f:
            json.dump(doubled, f)
        r = run_tool(self.base, self.out, 2)
        self.assertEqual(r.returncode, 2)
        self.assertIn("genau 1x", r.stderr)


if __name__ == "__main__":
    unittest.main()
