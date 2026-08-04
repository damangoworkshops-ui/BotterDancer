"""Vertragstests fuer Kontakt-Rekonstruktion + Root-Anchoring (Stufe 2)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from foot_core import (  # noqa: E402
    clean_contacts,
    contact_segments,
    detect_contacts,
    root_anchor_correction,
    skate_metric,
)

FPS = 30.0


class TestContacts(unittest.TestCase):
    def test_niedrig_und_langsam(self):
        y = np.array([0.2, 0.03, 0.03, 0.03, 0.2])
        v = np.array([1.0, 0.1, 0.1, 0.1, 1.0])
        c = detect_contacts(y, v, floor_y=0.0)
        self.assertTrue(np.array_equal(c, [False, True, True, True, False]))

    def test_schneller_durchschwung_ist_kein_kontakt(self):
        y = np.full(5, 0.03)
        v = np.array([0.1, 2.0, 2.0, 0.1, 0.1])  # tief, aber schnell
        c = detect_contacts(y, v, floor_y=0.0)
        self.assertTrue(np.array_equal(c, [True, False, False, True, True]))

    def test_segmente_und_flackern(self):
        mask = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
        self.assertEqual(contact_segments(mask, min_len=3), [(1, 4)])
        cleaned = clean_contacts(mask, min_len=3)
        self.assertFalse(cleaned[8])  # Einzel-Frame-Flackern (isoliert) entfernt
        self.assertTrue(cleaned[2])

    def test_luecken_werden_geschlossen(self):
        """Verifier-Befund 05.08.: 1-Frame-Dropout mitten im Kontakt darf den
        Anker nicht resetten — Luecke wird VOR der min_len-Filterung gefuellt."""
        mask = np.zeros(30, dtype=bool)
        mask[5:13] = True
        mask[14:22] = True  # 1-Frame-Loch bei t=13
        cleaned = clean_contacts(mask, min_len=3, close_gaps=2)
        self.assertTrue(cleaned[13])
        self.assertTrue(np.all(cleaned[5:22]))
        # Rand-Luecken (Anfang/Ende) werden NICHT gefuellt
        self.assertFalse(cleaned[0])
        self.assertFalse(cleaned[-1])


class TestRootAnchor(unittest.TestCase):
    def test_slide_wird_eliminiert(self):
        """Fuss rutscht waehrend Kontakt linear 10cm -> Korrektur haelt ihn planted."""
        n = 60
        feet = np.zeros((n, 1, 2))
        feet[:, 0, 0] = np.linspace(0.0, 0.10, n)  # konstanter Drift in x
        contacts = np.ones((n, 1), dtype=bool)
        corr = root_anchor_correction(feet, contacts, FPS)
        planted = feet[:, 0, :] + corr
        drift = np.linalg.norm(planted - planted[0], axis=-1)
        self.assertLess(drift.max(), 0.015)  # Rest nur durch Glaettung
        before = skate_metric(feet, contacts, FPS)
        after = skate_metric(feet + corr[:, None, :], contacts, FPS)
        self.assertLess(after, before * 0.2)

    def test_kein_kontakt_keine_aenderung_der_dynamik(self):
        """Ohne Kontakt haelt die Korrektur ihren letzten Wert (kein Snap-Back)."""
        n = 40
        feet = np.zeros((n, 1, 2))
        feet[:20, 0, 0] = np.linspace(0.0, 0.05, 20)
        contacts = np.zeros((n, 1), dtype=bool)
        contacts[:20, 0] = True
        corr = root_anchor_correction(feet, contacts, FPS)
        # nach Kontaktende: Korrektur konstant
        self.assertTrue(np.allclose(corr[25:], corr[30], atol=1e-9))

    def test_doppelkontakt_mittelt(self):
        n = 30
        feet = np.zeros((n, 2, 2))
        feet[:, 0, 0] = np.linspace(0.0, 0.04, n)    # Fuss A driftet +x
        feet[:, 1, 0] = np.linspace(0.0, -0.04, n)   # Fuss B driftet -x
        contacts = np.ones((n, 2), dtype=bool)
        corr = root_anchor_correction(feet, contacts, FPS)
        # Forderungen heben sich auf -> Korrektur bleibt nahe 0
        self.assertLess(np.abs(corr).max(), 0.005)

    def test_kein_entladesprung_bei_doppel_zu_einzel(self):
        """Verifier-Befund 05.08. (HOCH): symmetrisch divergierende Fuesse im
        Doppelkontakt, dann hebt einer ab — die akkumulierte Disparitaet darf
        sich NICHT als Ein-Frame-Sprung entladen."""
        n = 60
        feet = np.zeros((n, 2, 2))
        feet[:40, 0, 0] = np.linspace(0.0, 0.6, 40)     # A driftet stark +x
        feet[:40, 1, 0] = np.linspace(0.0, -0.6, 40)    # B driftet stark -x
        feet[40:, 1, 0] = feet[39, 1, 0]                # B bleibt, A hebt ab
        feet[40:, 0, 0] = feet[39, 0, 0]
        contacts = np.ones((n, 2), dtype=bool)
        contacts[40:, 0] = False                        # Doppel -> Einzel bei t=40
        corr = root_anchor_correction(feet, contacts, FPS, smooth_s=0.0)
        step = np.linalg.norm(np.diff(corr, axis=0), axis=-1)
        # vorher: Sprung von ~0.3 (halbe Disparitaet); jetzt: kein Schritt
        # groesser als der pro-Frame-Drift (0.6/40 = 1.5cm) + Toleranz
        self.assertLess(step.max(), 0.02, f"max step {step.max():.3f}")

    def test_neuer_kontakt_ankert_an_korrigierter_position(self):
        """Kettenverhalten: zweiter Kontakt darf die Korrektur des ersten nicht
        rueckgaengig machen, sondern baut auf ihr auf."""
        n = 60
        feet = np.zeros((n, 1, 2))
        feet[:25, 0, 0] = np.linspace(0.0, 0.06, 25)          # Kontakt 1 rutscht
        feet[25:35, 0, 0] = np.linspace(0.06, 0.30, 10)        # Schritt (Flug)
        feet[35:, 0, 0] = 0.30 + np.linspace(0.0, 0.06, 25)    # Kontakt 2 rutscht
        contacts = np.zeros((n, 1), dtype=bool)
        contacts[:25, 0] = True
        contacts[35:, 0] = True
        corr = root_anchor_correction(feet, contacts, FPS)
        planted2 = feet[38:, 0, 0] + corr[38:, 0]
        self.assertLess(planted2.max() - planted2.min(), 0.02)
        before = skate_metric(feet, contacts, FPS)
        after = skate_metric(feet + corr[:, None, :], contacts, FPS)
        self.assertLess(after, before * 0.35)


class TestSkateMetric(unittest.TestCase):
    def test_perfekte_plantung_ist_null(self):
        """Verifier-Befund 05.08.: Toe-off-Intervall zaehlte als Kontakt-Skate —
        perfekte Plantung ergab nie 0. Jetzt: beide Endpunkte im Kontakt."""
        n = 30
        feet = np.zeros((n, 1, 2))
        feet[20:, 0, 0] = np.linspace(0.0, 1.0, 10)  # Swing nach Toe-off
        contacts = np.zeros((n, 1), dtype=bool)
        contacts[5:20, 0] = True                      # planted, dann Abflug
        self.assertEqual(skate_metric(feet, contacts, FPS), 0.0)


if __name__ == "__main__":
    unittest.main()
