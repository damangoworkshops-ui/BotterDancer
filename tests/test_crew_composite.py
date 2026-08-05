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


def synth_clip(path, x_start, color, n=30, w=320, h=180, fps=16):
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
    subprocess.run([FFMPEG, "-v", "error", "-framerate", str(fps), "-i",
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

    def test_fps_konflikt_wird_abgelehnt(self):
        """Regression 06.08.: 30er-Raum-Plate + 16er-Figuren ergab einen 1.9x zu
        schnellen Clip — das faellt sonst erst im RIFE-Preflight auf."""
        a = os.path.join(self.d, "base30.mp4")
        b = os.path.join(self.d, "ov16.mp4")
        synth_clip(a, 20, (40, 40, 200), fps=30)
        synth_clip(b, 200, (40, 200, 40), fps=16)
        r = subprocess.run([sys.executable, TOOL, "--base", a, "--overlay", b,
                            "--out", os.path.join(self.d, "o.mp4")],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 2)
        self.assertIn("fps", r.stderr)

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


@unittest.skipUnless(HAVE_CV2 and FFMPEG, "opencv/ffmpeg nicht verfuegbar")
class TestOverlayCrop(unittest.TestCase):
    """Regressionen aus dem Fable-Review 06.08. am --overlay-crop-Pfad."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _crop_setup(self, n=12):
        """Basis-Plate 128x128 (230), Crop-Clip 64x96 (BG 200, wandernder
        dunkler Block), Sidecar + Crop-Pose-PNGs fuer den Massstabs-Fallback."""
        import json
        base = os.path.join(self.d, "base.mp4")
        bd = base + "_f"
        os.makedirs(bd)
        for i in range(n):
            cv2.imwrite(os.path.join(bd, "f%04d.png" % i),
                        np.full((128, 128, 3), 230, np.uint8))
        subprocess.run([FFMPEG, "-v", "error", "-framerate", "16", "-i",
                        os.path.join(bd, "f%04d.png"), "-c:v", "libx264", "-crf", "8",
                        "-pix_fmt", "yuv420p", "-y", base],
                       check=True, capture_output=True)
        ov = os.path.join(self.d, "crop.mp4")
        od = ov + "_f"
        os.makedirs(od)
        for i in range(n):
            img = np.full((96, 64, 3), 200, np.uint8)
            x = 4 + 3 * i
            cv2.rectangle(img, (x, 24), (x + 16, 72), (40, 40, 40), -1)
            cv2.imwrite(os.path.join(od, "f%04d.png" % i), img)
        subprocess.run([FFMPEG, "-v", "error", "-framerate", "16", "-i",
                        os.path.join(od, "f%04d.png"), "-c:v", "libx264", "-crf", "8",
                        "-pix_fmt", "yuv420p", "-y", ov],
                       check=True, capture_output=True)
        pose_dir = os.path.join(self.d, "cropdir")
        os.makedirs(pose_dir)
        for i in range(n):
            p = np.zeros((96, 64, 3), np.uint8)
            cv2.line(p, (32, 30), (32, 65), (0, 255, 0), 3)   # Skelett 35 px
            cv2.imwrite(os.path.join(pose_dir, "pose_%04d.png" % i), p)
        sidecar = pose_dir + ".crop.json"
        with open(sidecar, "w") as f:
            json.dump({"canvas": [128, 128], "out": [64, 96],
                       "windows": [[32.0, 16.0, 64.0, 96.0]] * n,
                       "source_pose_dir": pose_dir, "pixel_gain": 1.0}, f)
        return base, ov, sidecar

    def test_helligkeitsangleich_brennt_crop_nicht_aus(self):
        """Regression: ov_bg wurde NACH uncrop gemessen — Median ueber die
        mehrheitlich schwarze Leinwand = 0, shift ~ +230, Figur weiss."""
        base, ov, sidecar = self._crop_setup()
        out = os.path.join(self.d, "out.mp4")
        r = subprocess.run([sys.executable, TOOL, "--base", base, "--overlay", ov,
                            "--overlay-crop", sidecar, "--out", out],
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        cap = cv2.VideoCapture(out)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        # Figur bei Frame 6: Fenster (32,16) + Block x=4+18 -> Canvas ~ (54..70, 40..88)
        fig = frame[50:80, 56:68].astype(float)
        self.assertLess(fig.mean(), 150,
                        f"Crop-Figur ausgebrannt (Mittel {fig.mean():.0f}) — "
                        f"Helligkeitsreferenz wieder auf der leeren Leinwand gemessen?")

    def test_massstab_fallback_nutzt_crop_posen(self):
        """Regression: der Fallback las source_pose_dir (Vollbild-Skelette) und
        rechnete in einem fremden Koordinatenraum — Figur ~40% zu klein."""
        from crew_composite import _scale_correction_ratio, WAN_SCALE_BIAS
        _, _, sidecar = self._crop_setup()
        import json
        with open(sidecar) as f:
            cm = json.load(f)
        cm["crop_pose_dir"] = cm["source_pose_dir"]     # wie main() es ableitet
        frames = np.full((12, 96, 64, 3), 200, np.uint8)
        frames[:, 24:72, 20:40] = 40                    # Figur ~47 px hoch
        sc = _scale_correction_ratio(frames, cm)
        # Skelett ~37 px, Figur ~47 px: mit Bias ~1.07 — die alte Fassung ohne
        # Bias lieferte ~0.79 und haette die Figur zu klein gesetzt.
        self.assertGreaterEqual(sc, 0.95)
        self.assertLessEqual(sc, 1.2)
        self.assertGreater(WAN_SCALE_BIAS, 1.0)

    def test_massstab_fallback_ohne_posen_ist_neutral(self):
        """Fehlt das Crop-Posen-Verzeichnis, darf nicht still skaliert werden."""
        from crew_composite import _scale_correction_ratio
        frames = np.full((6, 96, 64, 3), 200, np.uint8)
        sc = _scale_correction_ratio(frames, {"canvas": [128, 128], "out": [64, 96],
                                              "windows": [[0, 0, 64, 96]] * 6})
        self.assertEqual(sc, 1.0)


if __name__ == "__main__":
    unittest.main()
