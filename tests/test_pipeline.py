"""Vertragstests fuer den Pipeline-Runner: Rezept-Validierung und Modus-Kopplung.

Kernpunkt: camera=moving KANN keine Gruppen (GVHMR trackt eine Person). Diese
Kopplung muss VOR der GPU-Zeit auffallen, nicht mittendrin.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "tools"))
TOOL = os.path.join(PROJECT, "tools", "pipeline.py")
from pipeline import STEPS, validate  # noqa: E402

COMFY_IN = r"C:\ComfyUI\input"


def spec(**over):
    s = {"name": "t", "source_video": __file__, "camera": "static",
         "cast": [{"ref": "__ref_a.png", "prompt": "a dancer"}]}
    s.update(over)
    return s


class TestValidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = []
        for n in ("__ref_a.png", "__ref_b.png"):
            p = os.path.join(COMFY_IN, n)
            if os.path.isdir(COMFY_IN) and not os.path.exists(p):
                open(p, "wb").close()
                cls.refs.append(p)

    @classmethod
    def tearDownClass(cls):
        for p in cls.refs:
            try:
                os.unlink(p)
            except OSError:
                pass

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_gueltiges_rezept(self):
        self.assertEqual(validate(spec()), [])

    def test_pflichtfelder(self):
        errs = validate({"cast": []})
        self.assertTrue(any("name" in e for e in errs))
        self.assertTrue(any("camera" in e for e in errs))
        self.assertTrue(any("cast" in e for e in errs))

    def test_unbekannter_kameramodus(self):
        errs = validate(spec(camera="drone"))
        self.assertTrue(any("static" in e and "moving" in e for e in errs), errs)

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_moving_erlaubt_keine_gruppe(self):
        """Die zentrale Kopplung: freie Kamera => nur eine Person."""
        errs = validate(spec(camera="moving",
                             cast=[{"ref": "__ref_a.png", "prompt": "a"},
                                   {"ref": "__ref_b.png", "prompt": "b"}]))
        self.assertTrue(any("GVHMR" in e and "static" in e for e in errs), errs)

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_moving_mit_einer_figur_ok(self):
        self.assertEqual(validate(spec(camera="moving")), [])

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_static_erlaubt_gruppe(self):
        errs = validate(spec(cast=[{"ref": "__ref_a.png", "prompt": "a"},
                                   {"ref": "__ref_b.png", "prompt": "b"}]))
        self.assertEqual(errs, [])

    def test_fehlende_referenzdatei(self):
        errs = validate(spec(cast=[{"ref": "gibt_es_nicht_xyz.png", "prompt": "a"}]))
        self.assertTrue(any("nicht in" in e for e in errs), errs)

    def test_unbekannte_qualitaet(self):
        errs = validate(spec(quality="ultra"))
        self.assertTrue(any("quality" in e for e in errs), errs)

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_background_room_ist_erlaubt(self):
        self.assertEqual(validate(spec(background="room")), [])

    def test_unbekannter_hintergrund(self):
        errs = validate(spec(background="beach"))
        self.assertTrue(any("background" in e for e in errs), errs)

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_identity_gate_werte(self):
        for g in ("off", "warn", "strict"):
            self.assertEqual(validate(spec(identity_gate=g)), [], g)
        errs = validate(spec(identity_gate="maybe"))
        self.assertTrue(any("identity_gate" in e for e in errs), errs)

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_identity_schwelle_muss_plausibel_sein(self):
        self.assertEqual(validate(spec(identity_min_sim=0.6)), [])
        for bad in (0.0, 1.0, 1.5, "hoch"):
            errs = validate(spec(identity_min_sim=bad))
            self.assertTrue(any("identity_min_sim" in e for e in errs), f"{bad}: {errs}")

    @unittest.skipUnless(os.path.isdir(COMFY_IN), "ComfyUI-Input fehlt")
    def test_room_braucht_statische_kamera(self):
        """Das Raum-Plate stammt aus einer statischen Kamera — eine virtuelle
        Fahrt darueber haette keine passende Parallaxe."""
        errs = validate(spec(background="room", camera="moving"))
        self.assertTrue(any("static" in e for e in errs), errs)


class TestCli(unittest.TestCase):
    def test_dry_run_plant_ohne_auszufuehren(self):
        d = tempfile.mkdtemp()
        jp = os.path.join(d, "job.json")
        with open(jp, "w") as f:
            json.dump(spec(name="drytest"), f)
        r = subprocess.run([sys.executable, TOOL, "--job", jp, "--dry-run",
                            "--stop-after", "segment"],
                           capture_output=True, text=True, timeout=120)
        if "FEHLER" in r.stderr and "nicht in" in r.stderr:
            self.skipTest("Referenzbild nicht vorhanden")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DRY-RUN", r.stdout)

    def test_ungueltiges_rezept_exit_2(self):
        d = tempfile.mkdtemp()
        jp = os.path.join(d, "bad.json")
        with open(jp, "w") as f:
            json.dump({"name": "x"}, f)
        r = subprocess.run([sys.executable, TOOL, "--job", jp],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertIn("FEHLER", r.stderr)

    def test_schrittliste_vollstaendig(self):
        self.assertEqual(STEPS[0], "segment")
        self.assertEqual(STEPS[-1], "export")
        self.assertIn("composite", STEPS)


if __name__ == "__main__":
    unittest.main()
