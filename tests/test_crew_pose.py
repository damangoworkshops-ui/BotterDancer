"""Vertragstests fuer Multi-Person-Tracking (Full-Crew-Casting).

Kernrisiko: wechseln Tracks die Identitaet, rendert Wan-Animate Figuren, die
ineinander morphen — genau das, was das Casting verhindern soll.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

HAVE_AUX = os.path.isdir(r"C:\ComfyUI\custom_nodes\comfyui_controlnet_aux\src")
if HAVE_AUX:
    from crew_pose import anchor, bbox_area, build_tracks, fill_gaps

W, H = 1024, 576


def person(cx, cy=300, scale=1.0):
    """OpenPose-18-Person um (cx, cy); scale skaliert die Bounding-Box."""
    pts = [(cx, cy - 90), (cx, cy - 70),
           (cx - 20, cy - 70), (cx - 28, cy - 40), (cx - 30, cy - 10),
           (cx + 20, cy - 70), (cx + 28, cy - 40), (cx + 30, cy - 10),
           (cx - 12, cy), (cx - 14, cy + 45), (cx - 15, cy + 90),
           (cx + 12, cy), (cx + 14, cy + 45), (cx + 15, cy + 90),
           (cx - 4, cy - 94), (cx + 4, cy - 94), (cx - 9, cy - 91), (cx + 9, cy - 91)]
    flat = []
    for x, y in pts:
        flat += [cx + (x - cx) * scale, cy + (y - cy) * scale, 1.0]
    return {"pose_keypoints_2d": flat}


def frame(*people):
    return {"people": list(people), "canvas_width": W, "canvas_height": H}


@unittest.skipUnless(HAVE_AUX, "comfyui_controlnet_aux nicht vorhanden")
class TestTracking(unittest.TestCase):
    def test_stabile_zuordnung_links_nach_rechts(self):
        frames = [frame(person(200 + i), person(500 + i), person(800 + i)) for i in range(20)]
        tracks = build_tracks(frames, 3)
        for t, expected_x in enumerate((200, 500, 800)):
            xs = [anchor(tracks[t][i])[0] for i in range(20)]
            self.assertAlmostEqual(xs[0], expected_x, delta=25)
            self.assertTrue(np.all(np.diff(xs) > 0))  # jeder Track folgt SEINER Person

    def test_reihenfolge_der_eingabe_egal(self):
        """DWPose liefert Personen in beliebiger Reihenfolge — Tracks duerfen
        davon nicht abhaengen."""
        frames = []
        for i in range(15):
            ppl = [person(200 + i), person(500 + i), person(800 + i)]
            frames.append(frame(*(ppl[::-1] if i % 2 else ppl)))  # jede 2. Frame gedreht
        tracks = build_tracks(frames, 3)
        for t in range(3):
            xs = np.array([anchor(tracks[t][i])[0] for i in range(15)])
            self.assertLess(np.abs(np.diff(xs)).max(), 30, f"Track {t} springt: {xs}")

    def test_annaeherung_ohne_identitaetstausch(self):
        """Zwei Taenzerinnen laufen aufeinander zu und wieder auseinander —
        die Tracks duerfen sich nicht vertauschen."""
        frames = []
        for i in range(24):
            d = 300 - 10 * min(i, 12) + 10 * max(0, i - 12)  # 300 -> 180 -> 300
            frames.append(frame(person(512 - d // 2), person(512 + d // 2)))
        tracks = build_tracks(frames, 2)
        left = np.array([anchor(tracks[0][i])[0] for i in range(24)])
        right = np.array([anchor(tracks[1][i])[0] for i in range(24)])
        self.assertTrue(np.all(left < right), "Tracks haben die Seite getauscht")

    def test_hintergrundperson_wird_verworfen(self):
        """Nur die groessten --crew Personen zaehlen (Zuschauer/Spiegelbild raus)."""
        frames = [frame(person(200), person(800), person(950, cy=200, scale=0.3))
                  for _ in range(10)]
        tracks = build_tracks(frames, 2)
        for t in range(2):
            for i in range(10):
                self.assertGreater(bbox_area(tracks[t][i]), 5000)

    def test_luecken_interpolation(self):
        track = [person(100), None, None, person(160)]
        n_i, n_h = fill_gaps(track, max_gap=8)
        self.assertEqual((n_i, n_h), (2, 0))
        xs = [anchor(p)[0] for p in track]
        self.assertTrue(np.all(np.diff(xs) > 0))
        self.assertAlmostEqual(xs[1], 120, delta=2)

    def test_randluecke_wird_gehalten(self):
        track = [None, None, person(300), person(310)]
        n_i, n_h = fill_gaps(track, max_gap=8)
        self.assertEqual((n_i, n_h), (0, 2))
        self.assertIsNotNone(track[0])
        self.assertAlmostEqual(anchor(track[0])[0], 300, delta=2)


if __name__ == "__main__":
    unittest.main()
