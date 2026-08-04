"""Vertragstests fuer die One-Shot-Kameratrajektorien."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from cam_trajectory import smoothstep, trajectory  # noqa: E402


class TestTrajectory(unittest.TestCase):
    def test_statisch_ohne_end(self):
        t = trajectory(80, 90.0)
        self.assertEqual(t.shape, (80,))
        self.assertTrue(np.all(t == 90.0))

    def test_endpunkte_exakt(self):
        t = trajectory(150, -50.0, 85.0)
        self.assertAlmostEqual(t[0], -50.0)
        self.assertAlmostEqual(t[-1], 85.0)

    def test_monoton_und_geeast(self):
        t = trajectory(100, 0.0, 10.0)
        self.assertTrue(np.all(np.diff(t) >= 0))
        # smoothstep: langsam anfahren -> erstes Drittel legt weniger zurueck als Mitte
        self.assertLess(t[33] - t[0], t[66] - t[33])

    def test_linear(self):
        t = trajectory(11, 0.0, 10.0, easing="linear")
        self.assertTrue(np.allclose(t, np.arange(11.0)))

    def test_rueckwaerts(self):
        t = trajectory(50, 1.0, 0.5)
        self.assertTrue(np.all(np.diff(t) <= 0))
        self.assertAlmostEqual(t[-1], 0.5)

    def test_smoothstep_ableitung_null_an_enden(self):
        u = np.linspace(0, 1, 1001)
        s = smoothstep(u)
        self.assertLess(s[1] - s[0], 1e-5)      # Anfahren ohne Ruck
        self.assertLess(s[-1] - s[-2], 1e-5)    # Ausrollen ohne Ruck

    def test_ungueltige_argumente(self):
        with self.assertRaises(ValueError):
            trajectory(0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            trajectory(10, 0.0, 1.0, easing="bounce")


if __name__ == "__main__":
    unittest.main()
