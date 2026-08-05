"""Vertragstests fuer die Naht-Metrik: Sie muss einen echten Chunk-Sprung
FINDEN und darf bei normaler Bewegung NICHT anschlagen (beide Richtungen —
ein Gate, das nur eine Seite kann, erzeugt falsche Sicherheit; Lehre aus dem
verworfenen Farb-Drift-Gate vom 04.08.)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(PROJECT, "tools", "seam_check.py")
FFMPEG_FALLBACK = (r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages"
                   r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                   r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe")
FFMPEG = shutil.which("ffmpeg") or FFMPEG_FALLBACK
HAVE_FFMPEG = os.path.isfile(FFMPEG) if not shutil.which("ffmpeg") else True

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def synth_clip(path, n=60, seam=None, jump_px=40, color_shift=0):
    """Graue Buehne + wandernder farbiger Block (= 'Figur'), konstante Schrittweite.
    seam: ab diesem Frame springt die Figur (Bewegungssprung) und/oder aendert
    ihre Farbe (Identitaetssprung)."""
    import cv2
    frames = []
    for i in range(n):
        img = np.full((192, 128, 3), 200, np.uint8)  # heller Studio-BG
        # Pendel (Dreieck) statt Modulo-Saegezahn: konstante Schrittweite OHNE
        # Positionssprung. Die erste Fixture-Version hatte einen %-Wrap bei
        # Frame 60 — die Metrik hat ihn korrekt als Sprung gemeldet (06.08.).
        x = 20 + abs((i % 80) - 40)
        if seam is not None and i >= seam:
            x += jump_px
        color = (60, 90, 220)
        if seam is not None and i >= seam and color_shift:
            color = (220, 90, 60)  # deutlich andere Figur-Farbe
        cv2.rectangle(img, (x, 60), (x + 30, 150), color, -1)
        frames.append(img)
    tmp = path + "_frames"
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, "f%04d.png" % i), f)
    subprocess.run([FFMPEG, "-v", "error", "-framerate", "16", "-i",
                    os.path.join(tmp, "f%04d.png"), "-c:v", "libx264", "-crf", "10",
                    "-pix_fmt", "yuv420p", "-y", path], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_FFMPEG and HAVE_CV2, "ffmpeg/opencv nicht verfuegbar")
class TestSeamCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _run(self, video, seam):
        return subprocess.run([sys.executable, TOOL, "--video", video, "--seams", str(seam)],
                              capture_output=True, text=True, timeout=120)

    def test_saubere_naht_schlaegt_nicht_an(self):
        v = os.path.join(self.d, "clean.mp4")
        synth_clip(v, n=60, seam=None)
        r = self._run(v, 30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("unauffaellig", r.stdout)

    def test_bewegungssprung_wird_gefunden(self):
        v = os.path.join(self.d, "jump.mp4")
        synth_clip(v, n=60, seam=30, jump_px=40)
        r = self._run(v, 30)
        self.assertEqual(r.returncode, 4, r.stdout)
        self.assertIn("BEWEGUNGSSPRUNG", r.stdout)

    def test_identitaetssprung_wird_gefunden(self):
        v = os.path.join(self.d, "ident.mp4")
        synth_clip(v, n=60, seam=30, jump_px=0, color_shift=1)
        r = self._run(v, 30)
        self.assertEqual(r.returncode, 4, r.stdout)
        self.assertIn("IDENTITAETSSPRUNG", r.stdout)

    def test_naht_ausserhalb_wird_uebersprungen(self):
        v = os.path.join(self.d, "short.mp4")
        synth_clip(v, n=40, seam=None)
        r = subprocess.run([sys.executable, TOOL, "--video", v, "--seams", "500"],
                           capture_output=True, text=True, timeout=120)
        self.assertIn("ausserhalb", r.stderr)

    def test_pose_kontrolle_entlastet_bei_input_sprung(self):
        """Kernvertrag der Kontrollmessung (06.08. am echten Chain-Render
        entwickelt): springt schon der POSE-INPUT an der Naht, ist der Render
        unschuldig — dann darf das Gate NICHT anschlagen."""
        import cv2
        v = os.path.join(self.d, "inputjump.mp4")
        synth_clip(v, n=60, seam=30, jump_px=40)      # Video springt...
        pose = os.path.join(self.d, "posedir")
        os.makedirs(pose, exist_ok=True)
        for i in range(60):                            # ...und der Input auch
            img = np.zeros((192, 128, 3), np.uint8)
            x = 20 + abs((i % 80) - 40) + (40 if i >= 30 else 0)
            cv2.rectangle(img, (x, 60), (x + 30, 150), (255, 255, 255), -1)
            cv2.imwrite(os.path.join(pose, "p%04d.png" % i), img)
        r = subprocess.run([sys.executable, TOOL, "--video", v, "--seams", "30",
                            "--pose-dir", pose], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Render-Ueberschuss", r.stdout)

    def test_pose_kontrolle_meldet_reines_render_artefakt(self):
        """Umkehrung: Input glatt, Video springt -> Artefakt muss gemeldet werden."""
        import cv2
        v = os.path.join(self.d, "renderjump.mp4")
        synth_clip(v, n=60, seam=30, jump_px=40)
        pose = os.path.join(self.d, "posedir2")
        os.makedirs(pose, exist_ok=True)
        for i in range(60):                            # Input OHNE Sprung
            img = np.zeros((192, 128, 3), np.uint8)
            x = 20 + abs((i % 80) - 40)
            cv2.rectangle(img, (x, 60), (x + 30, 150), (255, 255, 255), -1)
            cv2.imwrite(os.path.join(pose, "p%04d.png" % i), img)
        r = subprocess.run([sys.executable, TOOL, "--video", v, "--seams", "30",
                            "--pose-dir", pose], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 4, r.stdout)
        self.assertIn("RENDER-NAHT-ARTEFAKT", r.stdout)

    def test_chunk_parameter_leiten_naehte_ab(self):
        v = os.path.join(self.d, "chunks.mp4")
        synth_clip(v, n=90, seam=None)
        r = subprocess.run([sys.executable, TOOL, "--video", v,
                            "--chunk-length", "30", "--chunks", "3"],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("[30, 60]", r.stdout)


if __name__ == "__main__":
    unittest.main()
