"""Integrationstests fuer filter_pose_v2: Synthetik-Gate (F5), transaktionale
Publikation + Lock (F4). Braucht das lokale ComfyUI-Setup (custom_controlnet_aux)
— wird ohne dieses uebersprungen. Fixtures sind rein synthetisch."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(PROJECT, "tools", "filter_pose_v2.py")
AUX = r"C:\ComfyUI\custom_nodes\comfyui_controlnet_aux\src"


def synth_person(cx=256.0, cy=380.0):
    """Plausible OpenPose-18-Person um (cx, cy), alle Confidences 1.0."""
    pts = [
        (cx, cy - 180), (cx, cy - 150),                      # Nase, Neck
        (cx - 40, cy - 150), (cx - 55, cy - 90), (cx - 60, cy - 30),   # R Arm
        (cx + 40, cy - 150), (cx + 55, cy - 90), (cx + 60, cy - 30),   # L Arm
        (cx - 25, cy), (cx - 28, cy + 90), (cx - 30, cy + 180),        # R Bein
        (cx + 25, cy), (cx + 28, cy + 90), (cx + 30, cy + 180),        # L Bein
        (cx - 8, cy - 188), (cx + 8, cy - 188),              # Augen
        (cx - 18, cy - 182), (cx + 18, cy - 182),            # Ohren
    ]
    flat = []
    for x, y in pts:
        flat += [x, y, 1.0]
    return {"pose_keypoints_2d": flat}


def synth_frames(n, person_at=None):
    """n Frames; person_at=None -> ueberall eine Person, sonst nur an den Indizes."""
    frames = []
    for i in range(n):
        has = person_at is None or i in person_at
        frames.append({"people": [synth_person()] if has else [],
                       "canvas_width": 512, "canvas_height": 768})
    return frames


@unittest.skipUnless(os.path.isdir(AUX), "comfyui_controlnet_aux nicht vorhanden")
class TestFilterPose(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, "pose.json")
        self.out = os.path.join(self.d, "outdir")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _run(self, frames, extra=()):
        with open(self.src, "w") as f:
            json.dump(frames, f)
        return subprocess.run(
            [sys.executable, TOOL, "--src", self.src, "--outdir", self.out, *extra],
            capture_output=True, text=True, timeout=300)

    def test_normalfall_transaktional(self):
        r = self._run(synth_frames(12))
        self.assertEqual(r.returncode, 0, r.stderr)
        pngs = [e for e in os.listdir(self.out) if e.endswith(".png")]
        self.assertEqual(len(pngs), 12)
        self.assertTrue(os.path.isfile(self.out + ".meta.json"))
        self.assertFalse(os.path.exists(self.out + ".lock"))
        self.assertFalse([e for e in os.listdir(self.d) if ".tmp-" in e])

    def test_synthetik_gate(self):
        """Nur 3/12 Frames echt -> 9 synthetisch (75%) > 30% -> Abbruch Exit 2,
        und ein VORHANDENER alter Output ueberlebt (F4+F5 zusammen)."""
        os.makedirs(self.out)
        marker = os.path.join(self.out, "pose_9999.png")
        open(marker, "wb").close()
        r = self._run(synth_frames(12, person_at={0, 5, 11}))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("synthetisch", r.stderr)
        self.assertTrue(os.path.exists(marker), "alter Output wurde angetastet!")

    def test_lock_blockiert(self):
        open(self.out + ".lock", "wb").close()
        r = self._run(synth_frames(4))
        self.assertEqual(r.returncode, 3)
        self.assertIn("Lock", r.stderr)


if __name__ == "__main__":
    unittest.main()
