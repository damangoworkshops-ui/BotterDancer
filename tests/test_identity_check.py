"""Vertragstests fuer den CLIP-Identitaets-Detektor.

Das eigentliche Diskriminierungs-Ergebnis ist am echten Material belegt und im
Docstring/Commit festgehalten (korrektes cam180 0.870 vs. falsche Variante
0.716 vs. fremde Referenz 0.412) — hier wird die Mechanik geprueft, die ohne
das 2.5-GB-Modell testbar ist.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "tools"))
TOOL = os.path.join(PROJECT, "tools", "identity_check.py")

try:
    import cv2
    HAVE_CV2 = True
    from identity_check import pose_box
except ImportError:
    HAVE_CV2 = False

HAVE_MODEL = os.path.isfile(r"C:\ComfyUI\models\clip_vision\clip_vision_h.safetensors")


@unittest.skipUnless(HAVE_CV2, "opencv nicht verfuegbar")
class TestPoseBox(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _pose(self, x0, y0, x1, y1, w=1024, h=576):
        img = np.zeros((h, w, 3), np.uint8)
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 6)
        p = os.path.join(self.d, "p.png")
        cv2.imwrite(p, img)
        return p

    def test_box_umschliesst_skelett_mit_rand(self):
        p = self._pose(400, 200, 500, 400)
        box = pose_box(p, (576, 1024))
        self.assertIsNotNone(box)
        x0, x1, y0, y1 = box
        self.assertLess(x0, 400)      # Rand nach aussen
        self.assertGreater(x1, 500)
        self.assertLess(y0, 200)
        self.assertGreater(y1, 400)

    def test_box_bleibt_im_bild(self):
        p = self._pose(5, 5, 20, 30)
        x0, x1, y0, y1 = pose_box(p, (576, 1024))
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 1024)
        self.assertLessEqual(y1, 576)

    def test_leeres_skelett_gibt_keine_box(self):
        img = np.zeros((576, 1024, 3), np.uint8)
        p = os.path.join(self.d, "empty.png")
        cv2.imwrite(p, img)
        self.assertIsNone(pose_box(p, (576, 1024)))

    def test_groessenanpassung(self):
        """Pose-PNG und Video koennen verschieden gross sein — die Box muss in
        Video-Koordinaten zurueckkommen."""
        p = self._pose(400, 200, 500, 400, w=1024, h=576)
        box = pose_box(p, (288, 512))   # halbe Groesse
        x0, x1, y0, y1 = box
        self.assertLessEqual(x1, 512)
        self.assertLessEqual(y1, 288)


@unittest.skipUnless(HAVE_CV2 and HAVE_MODEL, "clip_vision-Modell nicht vorhanden")
class TestDiscrimination(unittest.TestCase):
    """Ein Ende-zu-Ende-Lauf: identisches Bild muss ~1.0 ergeben, ein voellig
    anderes deutlich weniger. Faengt Vorzeichen-/Normierungsfehler im Embedding."""

    def test_selbstaehnlichkeit_und_kontrast(self):
        from identity_check import embed, load_clip_vision
        model, _ = load_clip_vision()
        rng = np.random.default_rng(3)
        a = np.full((256, 256, 3), 40, np.uint8)
        cv2.circle(a, (128, 128), 70, (30, 200, 240), -1)      # gelber Kreis
        b = np.full((256, 256, 3), 240, np.uint8)
        cv2.rectangle(b, (20, 20), (236, 236), (200, 30, 30), -1)  # blaues Quadrat
        va, vb = embed(model, a), embed(model, b)
        self.assertAlmostEqual(float(np.dot(va, va)), 1.0, places=4)  # normiert
        self.assertGreater(float(np.dot(va, embed(model, a.copy()))), 0.99)
        self.assertLess(float(np.dot(va, vb)), 0.9)


if __name__ == "__main__":
    unittest.main()
