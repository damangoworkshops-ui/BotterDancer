"""Vertragstest fuer das Sichtbarkeits-Gate in reproject_camera (Audit 04.08. F6).
Das Gate liegt inline in main() (voller Lauf braucht GVHMR-Daten); hier wird die
exakte Gate-Arithmetik als eigenstaendiger Vertrag festgeschrieben — weicht der
Code je davon ab, muss dieser Test bewusst mitgeaendert werden."""
import unittest

import numpy as np

THRESH_JOINTS = 4     # Frames mit weniger sichtbaren Joints gelten als leer
THRESH_SHARE = 0.2    # mehr als 20% leere Frames -> Abbruch (Exit 4)


def gate_fires(kp_valid, sel):
    vis = kp_valid[sel].sum(axis=1)
    n_low = int((vis < THRESH_JOINTS).sum())
    return n_low > THRESH_SHARE * len(sel), n_low


class TestVisibilityGate(unittest.TestCase):
    def test_degenerierte_sequenz_feuert(self):
        kp_valid = np.zeros((150, 17), bool)
        kp_valid[:, :2] = True  # ueberall nur 2 sichtbare Joints
        fires, n_low = gate_fires(kp_valid, list(range(80)))
        self.assertTrue(fires)
        self.assertEqual(n_low, 80)

    def test_gesunde_sequenz_schweigt(self):
        kp_valid = np.ones((150, 17), bool)
        fires, n_low = gate_fires(kp_valid, list(range(80)))
        self.assertFalse(fires)
        self.assertEqual(n_low, 0)

    def test_schwelle_exakt(self):
        kp_valid = np.ones((100, 17), bool)
        kp_valid[:20, :] = False  # exakt 20% leer -> feuert NICHT (>, nicht >=)
        fires, _ = gate_fires(kp_valid, list(range(100)))
        self.assertFalse(fires)
        kp_valid[20, :] = False   # 21% -> feuert
        fires, _ = gate_fires(kp_valid, list(range(100)))
        self.assertTrue(fires)


if __name__ == "__main__":
    unittest.main()
