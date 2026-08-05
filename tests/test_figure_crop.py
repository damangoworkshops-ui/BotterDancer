"""Vertragstests fuer Figuren-Crop und Rueckprojektion."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "tools"))
TOOL = os.path.join(PROJECT, "tools", "figure_crop.py")

try:
    import cv2
    HAVE_CV2 = True
    from crew_composite import uncrop
    from figure_crop import boxes_from_pose_pngs, plan_windows, smooth
except ImportError:
    HAVE_CV2 = False


@unittest.skipUnless(HAVE_CV2, "opencv nicht verfuegbar")
class TestWindowPlanning(unittest.TestCase):
    def test_fenster_hat_zielseitenverhaeltnis(self):
        boxes = [[400, 100, 500, 400]] * 20
        wins = plan_windows(boxes, 1024, 576, 512, 768)
        w, h = wins[0, 2], wins[0, 3]
        self.assertAlmostEqual(w / h, 512 / 768, places=2)

    def test_fenster_bleibt_im_canvas(self):
        boxes = [[0, 0, 80, 300]] * 10 + [[950, 300, 1024, 570]] * 10
        wins = plan_windows(boxes, 1024, 576, 512, 768)
        self.assertTrue(np.all(wins[:, 0] >= -0.01))
        self.assertTrue(np.all(wins[:, 1] >= -0.01))
        self.assertTrue(np.all(wins[:, 0] + wins[:, 2] <= 1024.01))
        self.assertTrue(np.all(wins[:, 1] + wins[:, 3] <= 576.01))

    def test_fenstergroesse_ist_konstant(self):
        """Konstante Groesse = konstante Skalierung; sonst pumpt die Figur."""
        boxes = [[400 + i, 100, 500 + i, 400 + (i % 5) * 10] for i in range(30)]
        wins = plan_windows(boxes, 1024, 576, 512, 768)
        self.assertLess(np.ptp(wins[:, 2]), 1e-6)
        self.assertLess(np.ptp(wins[:, 3]), 1e-6)

    def test_zentrum_wird_geglaettet(self):
        """Ein Ausreisser-Frame darf das Fenster nicht springen lassen."""
        boxes = [[400, 100, 500, 400]] * 20
        boxes[10] = [700, 100, 800, 400]
        wins = plan_windows(boxes, 1024, 576, 512, 768, smooth_win=9)
        steps = np.abs(np.diff(wins[:, 0]))
        self.assertLess(steps.max(), 60)

    def test_glaettung_erhaelt_laenge(self):
        self.assertEqual(len(smooth([1.0] * 15, 9)), 15)

    def test_leere_boxen(self):
        self.assertIsNone(plan_windows([None] * 5, 1024, 576, 512, 768))


@unittest.skipUnless(HAVE_CV2, "opencv nicht verfuegbar")
class TestBoxesFromPngs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_box_aus_skelett(self):
        p = os.path.join(self.d, "p.png")
        img = np.zeros((576, 1024, 3), np.uint8)
        cv2.line(img, (500, 200), (520, 400), (0, 255, 0), 5)
        cv2.imwrite(p, img)
        b = boxes_from_pose_pngs([p], margin=0.0)[0]
        self.assertIsNotNone(b)
        self.assertLessEqual(b[0], 500)
        self.assertGreaterEqual(b[2], 520)

    def test_leeres_png_gibt_none(self):
        p = os.path.join(self.d, "e.png")
        cv2.imwrite(p, np.zeros((576, 1024, 3), np.uint8))
        self.assertIsNone(boxes_from_pose_pngs([p])[0])


@unittest.skipUnless(HAVE_CV2, "opencv nicht verfuegbar")
class TestUncrop(unittest.TestCase):
    def test_inhalt_landet_im_fenster(self):
        crop = np.full((10, 768, 512, 3), 200, np.uint8)
        meta = {"windows": [[100.0, 50.0, 200.0, 300.0]] * 10, "canvas": [1024, 576]}
        out = uncrop(crop, meta, 1024, 576)
        self.assertGreater(int(out[0, 200, 200].mean()), 150)   # im Fenster
        self.assertEqual(int(out[0, 10, 10].mean()), 0)         # ausserhalb leer

    def test_skalierung_ist_fussbuendig(self):
        """scale<1 verkleinert nach oben, die Standflaeche bleibt — sonst
        schwebt die Figur oder versinkt im Boden."""
        crop = np.full((3, 768, 512, 3), 200, np.uint8)
        meta = {"windows": [[100.0, 100.0, 200.0, 300.0]] * 3, "canvas": [1024, 576]}
        full = uncrop(crop, meta, 1024, 576, 1.0)
        small = uncrop(crop, meta, 1024, 576, 0.5)
        def bottom(img):
            ys, _ = np.where(img.max(axis=2) > 50)
            return ys.max()
        self.assertAlmostEqual(bottom(full[0]), bottom(small[0]), delta=2)
        def top(img):
            ys, _ = np.where(img.max(axis=2) > 50)
            return ys.min()
        self.assertGreater(top(small[0]), top(full[0]) + 100)


if __name__ == "__main__":
    unittest.main()
