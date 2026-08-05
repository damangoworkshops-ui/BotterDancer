"""Vertragstests: Akzente aus 2D-Posen + Phasen-Fit-Segmentwahl."""
import os
import sys
import unittest

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "tools"))
from pose_accents import accents_from_frames, body_scale, speed_series  # noqa: E402
from song_beats import pick_segment  # noqa: E402

W, H = 1024, 576


def person(cx, cy=300, scale=1.0):
    pts = [(cx, cy - 90), (cx, cy - 70), (cx - 20, cy - 70), (cx - 28, cy - 40),
           (cx - 30, cy - 10), (cx + 20, cy - 70), (cx + 28, cy - 40), (cx + 30, cy - 10),
           (cx - 12, cy), (cx - 14, cy + 45), (cx - 15, cy + 90), (cx + 12, cy),
           (cx + 14, cy + 45), (cx + 15, cy + 90), (cx - 4, cy - 94), (cx + 4, cy - 94),
           (cx - 9, cy - 91), (cx + 9, cy - 91)]
    flat = []
    for x, y in pts:
        flat += [cx + (x - cx) * scale, cy + (y - cy) * scale, 1.0]
    return {"pose_keypoints_2d": flat}


def frames_from_x(xs):
    return [{"people": [person(x)], "canvas_width": W, "canvas_height": H} for x in xs]


class TestSpeed(unittest.TestCase):
    def test_koerpergroesse(self):
        s1 = body_scale(person(300, scale=1.0))
        s2 = body_scale(person(300, scale=2.0))
        self.assertAlmostEqual(s2 / s1, 2.0, delta=0.05)

    def test_geschwindigkeit_ist_groessennormiert(self):
        """Dieselbe Bewegung relativ zur Koerperlaenge muss dieselbe
        Geschwindigkeit ergeben — sonst haengt die Schwelle an Aufloesung
        und Kameradistanz."""
        klein = [{"people": [person(300 + i * 5, scale=1.0)], "canvas_width": W,
                  "canvas_height": H} for i in range(10)]
        gross = [{"people": [person(300 + i * 10, scale=2.0)], "canvas_width": W,
                  "canvas_height": H} for i in range(10)]
        a = speed_series(klein)[2:].mean()
        b = speed_series(gross)[2:].mean()
        self.assertAlmostEqual(a, b, delta=0.02)

    def test_fehlende_person_haelt_wert(self):
        fr = frames_from_x([100, 110, 120])
        fr.insert(2, {"people": [], "canvas_width": W, "canvas_height": H})
        s = speed_series(fr)
        self.assertEqual(len(s), 4)
        self.assertTrue(np.all(np.isfinite(s)))


class TestAccents(unittest.TestCase):
    def test_findet_stopps(self):
        fps = 16.0
        t = np.arange(80) / fps
        xs = 400 + 60 * np.sin(2 * np.pi * 1.0 * t)  # Umkehrpunkte = Stopps, 1 Hz
        acc = accents_from_frames(frames_from_x(xs), fps)
        self.assertGreaterEqual(len(acc), 6)
        gaps = np.diff(acc)
        self.assertAlmostEqual(float(np.median(gaps)), 0.5, delta=0.12)


class TestPhaseFit(unittest.TestCase):
    def _beats(self, period, n=200, offset=0.0):
        return offset + np.arange(n) * period

    def test_waehlt_segment_mit_passender_phase(self):
        """Zwei gleich stabile Kandidaten, aber nur bei einem liegen die
        Choreo-Akzente auf dem Beat — der muss gewinnen."""
        period = 0.5
        beats = self._beats(period, 400)
        downbeats = np.arange(0, 190, 10.0)      # Kandidaten alle 10 s
        accents = np.array([0.0, 0.5, 1.0, 1.5, 2.0])   # exakt auf dem Raster
        start, phase = pick_segment(beats, downbeats, 200.0, 6.0, accents)
        self.assertIsNotNone(phase)
        self.assertLess(phase, 0.05)
        # Ohne Akzente wird nur Stabilitaet gewertet -> kein Phasenwert
        s2, p2 = pick_segment(beats, downbeats, 200.0, 6.0)
        self.assertIsNone(p2)

    def test_versetzte_akzente_geben_schlechten_fit(self):
        period = 0.5
        beats = self._beats(period, 400)
        downbeats = np.arange(0, 190, 10.0)
        offbeat = np.array([0.25, 0.75, 1.25, 1.75])   # exakt zwischen den Beats
        _, phase = pick_segment(beats, downbeats, 200.0, 6.0, offbeat)
        self.assertGreater(phase, 0.35)

    def test_rueckgabe_ist_immer_ein_paar(self):
        beats = self._beats(0.5, 100)
        s, p = pick_segment(beats, np.array([0.0, 10.0]), 50.0, 6.0)
        self.assertIsInstance(s, float)


if __name__ == "__main__":
    unittest.main()
