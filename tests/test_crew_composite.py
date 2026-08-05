"""Vertragstests fuers Crew-Compositing (Median-Background-Freistellung)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "tools"))
TOOL = os.path.join(PROJECT, "tools", "crew_composite.py")

try:
    import cv2
    HAVE_CV2 = True
    from crew_composite import find_tool, person_mask
    FFMPEG = find_tool("ffmpeg")
except ImportError:
    HAVE_CV2 = False
    FFMPEG = None


def synth_clip(path, x_start, color, n=30, w=320, h=180):
    """Statischer heller Hintergrund + wandernder farbiger Block."""
    import cv2
    d = path + "_f"
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        img = np.full((h, w, 3), 210, np.uint8)
        img[0:20, :] = 190  # etwas Struktur im BG
        x = x_start + i
        cv2.rectangle(img, (x, 60), (x + 30, 140), color, -1)
        cv2.imwrite(os.path.join(d, "f%04d.png" % i), img)
    subprocess.run([FFMPEG, "-v", "error", "-framerate", "16", "-i",
                    os.path.join(d, "f%04d.png"), "-c:v", "libx264", "-crf", "8",
                    "-pix_fmt", "yuv420p", "-y", path], check=True, capture_output=True)
    shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAVE_CV2 and FFMPEG, "opencv/ffmpeg nicht verfuegbar")
class TestComposite(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_maske_findet_bewegte_figur(self):
        """Median-Background: die Figur wird gefunden, der Hintergrund nicht."""
        frames = []
        for i in range(30):
            img = np.full((180, 320, 3), 210, np.uint8)
            cv2.rectangle(img, (30 + i * 5, 60), (60 + i * 5, 140), (40, 40, 200), -1)
            frames.append(img)
        m = person_mask(np.array(frames), thresh=18.0, feather=0)
        self.assertGreater(m.mean(), 0.02)   # Figur gefunden
        self.assertLess(m.mean(), 0.25)      # aber nicht das halbe Bild
        # Maske sitzt auf der Figur, nicht daneben
        i = 10
        self.assertGreater(m[i, 100, 30 + i * 5 + 15], 0.5)
        self.assertLess(m[i, 10, 10], 0.5)

    def test_statische_szene_hat_leere_maske(self):
        """Ohne Bewegung gibt es keine Figur — Maske muss praktisch leer sein."""
        frames = np.array([np.full((180, 320, 3), 210, np.uint8) for _ in range(20)])
        m = person_mask(frames, thresh=18.0, feather=0)
        self.assertLess(m.mean(), 0.001)

    def test_zwei_figuren_landen_im_composite(self):
        a = os.path.join(self.d, "a.mp4")
        b = os.path.join(self.d, "b.mp4")
        out = os.path.join(self.d, "out.mp4")
        synth_clip(a, 20, (40, 40, 200))     # rot links
        synth_clip(b, 200, (40, 200, 40))    # gruen rechts
        r = subprocess.run([sys.executable, TOOL, "--base", a, "--overlay", b,
                            "--out", out], capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        cap = cv2.VideoCapture(out)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        # beide Bloecke muessen im Ergebnis sein
        red = frame[100, 20 + 10 + 15]
        green = frame[100, 200 + 10 + 15]
        self.assertGreater(int(red[2]), 120, f"rote Figur fehlt: {red}")
        self.assertGreater(int(green[1]), 120, f"gruene Figur fehlt: {green}")

    def test_groessenkonflikt_wird_abgelehnt(self):
        a = os.path.join(self.d, "a.mp4")
        b = os.path.join(self.d, "b_small.mp4")
        synth_clip(a, 20, (40, 40, 200))
        synth_clip(b, 20, (40, 200, 40), w=160, h=90)
        r = subprocess.run([sys.executable, TOOL, "--base", a, "--overlay", b,
                            "--out", os.path.join(self.d, "o.mp4")],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 2)
        self.assertIn("Basis", r.stderr)


if __name__ == "__main__":
    unittest.main()
