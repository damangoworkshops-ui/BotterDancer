"""Vertragstests fuer Room-Keeper-Werkzeuge (Silhouetten-Maske, Raum-Plate)."""
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
SIL = os.path.join(PROJECT, "tools", "pose_silhouette.py")
PLATE = os.path.join(PROJECT, "tools", "room_plate.py")

try:
    import cv2
    HAVE_CV2 = True
    from pose_silhouette import silhouette
    from room_plate import find_tool
    FFMPEG = find_tool("ffmpeg")
except ImportError:
    HAVE_CV2 = False
    FFMPEG = None


def person(cx, cy=280, s=1.0):
    pts = [(cx, cy - 90), (cx, cy - 70), (cx - 20, cy - 70), (cx - 28, cy - 40),
           (cx - 30, cy - 10), (cx + 20, cy - 70), (cx + 28, cy - 40), (cx + 30, cy - 10),
           (cx - 12, cy), (cx - 14, cy + 45), (cx - 15, cy + 90), (cx + 12, cy),
           (cx + 14, cy + 45), (cx + 15, cy + 90), (cx - 4, cy - 94), (cx + 4, cy - 94),
           (cx - 9, cy - 91), (cx + 9, cy - 91)]
    flat = []
    for x, y in pts:
        flat += [cx + (x - cx) * s, cy + (y - cy) * s, 1.0]
    return {"pose_keypoints_2d": flat}


@unittest.skipUnless(HAVE_CV2, "opencv nicht verfuegbar")
class TestSilhouette(unittest.TestCase):
    def test_deckt_person_ab_aber_nicht_das_bild(self):
        m = silhouette([person(300)], 1024, 576, limb=26, head=34, dilate=28)
        cover = (m > 127).mean()
        self.assertGreater(cover, 0.02, "Maske zu klein — Figur waere nicht ersetzbar")
        self.assertLess(cover, 0.35, "Maske zu gross — der Raum wuerde neu erfunden")
        # Maske sitzt auf der Figur, nicht daneben
        self.assertGreater(m[280, 300], 127)
        self.assertLess(m[280, 900], 127)

    def test_leere_pose_gibt_leere_maske(self):
        m = silhouette([], 512, 288, limb=20, head=24, dilate=10)
        self.assertEqual(int((m > 127).sum()), 0)

    def test_mehrere_personen_werden_alle_abgedeckt(self):
        m = silhouette([person(200), person(800)], 1024, 576, 26, 34, 28)
        self.assertGreater(m[280, 200], 127)
        self.assertGreater(m[280, 800], 127)
        self.assertLess(m[280, 512], 127)  # dazwischen frei

    def test_dilatation_vergroessert(self):
        a = silhouette([person(400)], 1024, 576, 26, 34, dilate=0)
        b = silhouette([person(400)], 1024, 576, 26, 34, dilate=40)
        self.assertGreater((b > 127).sum(), (a > 127).sum() * 1.3)

    def test_cli_schreibt_sequenz(self):
        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "pose.json")
            with open(src, "w") as f:
                json.dump([{"people": [person(300 + i)], "canvas_width": 512,
                            "canvas_height": 288} for i in range(6)], f)
            out = os.path.join(d, "masks")
            r = subprocess.run([sys.executable, SIL, "--src", src, "--outdir", out],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(len([x for x in os.listdir(out) if x.endswith(".png")]), 6)
            self.assertIn("Abdeckung", r.stdout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAVE_CV2 and FFMPEG, "opencv/ffmpeg nicht verfuegbar")
class TestRoomPlate(unittest.TestCase):
    def test_bewegtes_objekt_verschwindet(self):
        """Zeitlicher Median entfernt, was sich bewegt — das ist die ganze Idee.
        Grenze (empirisch 06.08.): steht die Person zu konstant, bleiben Geister."""
        d = tempfile.mkdtemp()
        try:
            frames_dir = os.path.join(d, "f")
            os.makedirs(frames_dir)
            for i in range(40):
                img = np.full((180, 320, 3), 200, np.uint8)
                img[20:40, :] = 120                       # feste Struktur (Raum)
                x = 10 + i * 7                            # wandert weit
                cv2.rectangle(img, (x, 80), (x + 25, 150), (30, 30, 200), -1)
                cv2.imwrite(os.path.join(frames_dir, "f%04d.png" % i), img)
            vid = os.path.join(d, "in.mp4")
            subprocess.run([FFMPEG, "-v", "error", "-framerate", "16", "-i",
                            os.path.join(frames_dir, "f%04d.png"), "-c:v", "libx264",
                            "-crf", "8", "-pix_fmt", "yuv420p", "-y", vid],
                           check=True, capture_output=True)
            out = os.path.join(d, "plate.mp4")
            r = subprocess.run([sys.executable, PLATE, "--video", vid, "--out", out,
                                "--frames", "10"], capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            cap = cv2.VideoCapture(out)
            ok, plate = cap.read()
            cap.release()
            self.assertTrue(ok)
            # Objekt ist BGR (30,30,200), Hintergrund (200,200,200) — unterscheidbar
            # nur im Blau-/Gruen-Kanal, NICHT im roten (beide 200).
            self.assertGreater(int(plate[110, 150, 0]), 150, "bewegtes Objekt noch im Plate")
            self.assertLess(abs(int(plate[30, 150, 0]) - 120), 40, "Raum-Struktur verloren")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_zu_kurzes_video_wird_abgelehnt(self):
        d = tempfile.mkdtemp()
        try:
            fd = os.path.join(d, "f")
            os.makedirs(fd)
            for i in range(4):
                cv2.imwrite(os.path.join(fd, "f%04d.png" % i),
                            np.full((90, 160, 3), 200, np.uint8))
            vid = os.path.join(d, "short.mp4")
            subprocess.run([FFMPEG, "-v", "error", "-framerate", "16", "-i",
                            os.path.join(fd, "f%04d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-y", vid], check=True, capture_output=True)
            r = subprocess.run([sys.executable, PLATE, "--video", vid,
                                "--out", os.path.join(d, "o.mp4")],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 3)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
