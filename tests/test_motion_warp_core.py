"""Vertragstests fuer die Motion-Time-Warp-Kernmathematik."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from motion_warp_core import (  # noqa: E402
    axis_angle_to_quat,
    build_warp_map,
    estimate_period,
    find_accents,
    quat_to_axis_angle,
    slerp,
    warp_time,
)


class TestAxisAngleQuat(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(7)
        aa = rng.uniform(-2.0, 2.0, (50, 21, 3))
        back = quat_to_axis_angle(axis_angle_to_quat(aa))
        # Rotationen vergleichen, nicht Vektoren: aa und back koennen sich um
        # 2*pi-Aequivalenz unterscheiden — hier sind Winkel < pi erzwungen,
        # also muss der Vektor selbst uebereinstimmen, solange |aa| < pi.
        mask = np.linalg.norm(aa, axis=-1) < np.pi
        self.assertGreater(mask.mean(), 0.5)
        self.assertTrue(np.allclose(back[mask], aa[mask], atol=1e-9))

    def test_null_rotation(self):
        self.assertTrue(np.allclose(axis_angle_to_quat(np.zeros(3)), [1, 0, 0, 0]))
        self.assertTrue(np.allclose(quat_to_axis_angle(np.array([1.0, 0, 0, 0])), np.zeros(3)))


class TestAccents(unittest.TestCase):
    def test_findet_stopps(self):
        fps = 30.0
        t = np.arange(150) / fps
        speed = 1.0 + 0.9 * np.sin(2 * np.pi * 2.0 * t)  # Minima alle 0.5s
        accents = find_accents(speed, fps)
        self.assertGreaterEqual(len(accents), 8)
        period = estimate_period(accents)
        self.assertAlmostEqual(period, 0.5, delta=0.05)

    def test_mindestabstand(self):
        """Zwei glatte Taeler 0.2s auseinander (< min_gap 0.25s) -> nur das
        tiefere bleibt. Realistische Gauss-Taeler statt Einzel-Frame-Spikes:
        der Box-Filter macht aus Spikes Plateaus mit mehrdeutigen Minima."""
        fps = 30.0
        t = np.arange(90) / fps
        speed = np.ones_like(t)
        speed -= 0.5 * np.exp(-((t - 1.0) ** 2) / (2 * 0.05 ** 2))   # Tal 1 @1.0s
        speed -= 0.8 * np.exp(-((t - 1.2) ** 2) / (2 * 0.05 ** 2))   # tieferes Tal @1.2s
        accents = find_accents(speed, fps, min_gap_s=0.25)
        self.assertEqual(len(accents), 1, accents)
        self.assertAlmostEqual(accents[0], 1.2, delta=0.07)

    def test_zu_wenige_akzente(self):
        self.assertIsNone(estimate_period(np.array([1.0, 2.0])))


class TestTatum(unittest.TestCase):
    def test_ueberspringende_akzente(self):
        """Choreo-Realitaet (04.08. am echten Tanz entdeckt): Akzente sitzen auf
        einem feinen Raster, ueberspringen aber Slots — Median-Intervalle liefern
        dann Phantasie-Tempi; der Tatum-Scan muss das echte Raster finden."""
        from motion_warp_core import estimate_tatum
        p_true, off_true = 1.0 / 3.0, 0.02
        slots = np.array([2, 3, 6, 7, 9, 10, 12, 14])  # unregelmaessig uebersprungen
        rng = np.random.default_rng(11)
        accents = off_true + slots * p_true + rng.normal(0, 0.01, len(slots))
        p, off, resid = estimate_tatum(accents)
        self.assertAlmostEqual(p, p_true, delta=0.02)
        self.assertLess(resid, 0.03)


class TestGridOffset(unittest.TestCase):
    def test_findet_phasenlage(self):
        from motion_warp_core import best_grid_offset
        p = 0.577
        true_off = 0.21
        rng = np.random.default_rng(9)
        accents = true_off + np.arange(8) * p + rng.normal(0, 0.01, 8)
        off, resid = best_grid_offset(accents, p)
        self.assertAlmostEqual(off, true_off, delta=0.02)
        self.assertLess(resid, 0.02)


class TestWarpMap(unittest.TestCase):
    def test_perfekt_passende_akzente(self):
        beats = np.arange(0.5, 5.0, 0.5)
        accents = beats + 0.05  # leicht daneben, gut innerhalb +/-15%
        anchors, matched, dropped = build_warp_map(accents, 5.0, beats)
        self.assertEqual(dropped, 0)
        self.assertGreaterEqual(matched, 8)
        # Akzente landen exakt auf Beats: warp_time(beat) == accent
        for b, a in zip(beats, accents):
            self.assertAlmostEqual(float(warp_time(anchors, b)), a, places=6)

    def test_steigungsschranke_verwirft(self):
        beats = np.arange(0.5, 5.0, 0.5)
        accents = np.array([0.5 + 0.2])  # braeuchte 40% Stauchung -> verworfen
        anchors, matched, dropped = build_warp_map(accents, 5.0, beats)
        self.assertEqual(matched, 0)
        self.assertEqual(dropped, 1)
        # Ohne Anker: Identitaets-Abbildung
        self.assertAlmostEqual(float(warp_time(anchors, 2.0)), 2.0, places=6)

    def test_monotonie_erzwungen(self):
        beats = np.arange(0.25, 6.0, 0.25)
        rng = np.random.default_rng(3)
        accents = np.sort(rng.uniform(0.2, 4.8, 12))
        anchors, _, _ = build_warp_map(accents, 5.0, beats)
        self.assertTrue(np.all(np.diff(anchors[:, 0]) > 0))
        self.assertTrue(np.all(np.diff(anchors[:, 1]) > 0))
        # Steigungen innerhalb der Schranke (ausser Endsegment mit Steigung 1)
        slopes = np.diff(anchors[:, 1]) / np.diff(anchors[:, 0])
        self.assertTrue(np.all(slopes > 0.849), slopes)
        self.assertTrue(np.all(slopes < 1.151), slopes)


class TestSlerp(unittest.TestCase):
    def test_endpunkte_und_mitte(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])           # Identitaet
        q1 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90 Grad um X
        self.assertTrue(np.allclose(slerp(q0, q1, 0.0), q0))
        self.assertTrue(np.allclose(slerp(q0, q1, 1.0), q1))
        mid = slerp(q0, q1, 0.5)  # 45 Grad um X
        want = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])
        self.assertTrue(np.allclose(mid, want, atol=1e-9))

    def test_kuerzester_bogen(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = -np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])  # negiert = gleiche Rotation
        mid = slerp(q0, q1, 0.5)
        # darf NICHT den langen Weg nehmen: Winkel zur Identitaet klein
        self.assertGreater(abs(mid[0]), 0.99)

    def test_batched_gelenke_mit_frame_u(self):
        """Regression 04.08.: (F,J,4)-Quats mit (F,)-u muessen broadcasten
        (Frame-weises u ueber alle Gelenke) — genau der body_pose-Fall."""
        rng = np.random.default_rng(5)
        aa = rng.uniform(-1.0, 1.0, (150, 21, 3))
        q = axis_angle_to_quat(aa)
        u = rng.uniform(0.0, 1.0, 150)
        out = slerp(q, np.roll(q, -1, axis=0), u)
        self.assertEqual(out.shape, (150, 21, 4))
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=-1), 1.0))

    def test_identische_quaternionen(self):
        q = np.array([[0.5, 0.5, 0.5, 0.5]] * 4)
        out = slerp(q, q, 0.3)
        self.assertTrue(np.allclose(out, q))


if __name__ == "__main__":
    unittest.main()
