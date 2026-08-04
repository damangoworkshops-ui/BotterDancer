"""Konsistenzpruefung + Rotation des C2PA-Schluesselmaterials (Audit 04.08. F10).
Arbeitet komplett in einem Temp-Verzeichnis — beruehrt NIE assets/keys/."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

try:
    import cryptography  # noqa: F401
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


@unittest.skipUnless(HAVE_CRYPTO, "cryptography nicht verfuegbar")
class TestBundleLifecycle(unittest.TestCase):
    def setUp(self):
        import make_c2pa_cert as mc
        self.mc = mc
        self.d = Path(tempfile.mkdtemp())
        self._old = mc.KEYS_DIR
        mc.KEYS_DIR = self.d
        self._argv = sys.argv

    def tearDown(self):
        self.mc.KEYS_DIR = self._old
        sys.argv = self._argv
        shutil.rmtree(self.d, ignore_errors=True)

    def _generate(self):
        sys.argv = ["make_c2pa_cert.py"]
        self.assertEqual(self.mc.main(), 0)

    def test_lebenszyklus(self):
        self.assertEqual(self.mc.bundle_state(), "empty")
        self._generate()
        self.assertEqual(self.mc.bundle_state(), "consistent")

        # No-Op-Wiederholung darf nichts veraendern
        before = (self.d / "c2pa_leaf.key").read_bytes()
        self._generate()
        self.assertEqual((self.d / "c2pa_leaf.key").read_bytes(), before)

        # Teilzustand: harter Abbruch ohne --force (Audit F10)
        (self.d / "c2pa_chain.pem").unlink()
        self.assertEqual(self.mc.bundle_state(), "partial")
        sys.argv = ["make_c2pa_cert.py"]
        self.assertEqual(self.mc.main(), 1)

        # --force rotiert und sichert Bestand nach .bak
        sys.argv = ["make_c2pa_cert.py", "--force"]
        self.assertEqual(self.mc.main(), 0)
        self.assertEqual(self.mc.bundle_state(), "consistent")
        self.assertTrue((self.d / "c2pa_leaf.key.bak").exists())
        self.assertNotEqual((self.d / "c2pa_leaf.key").read_bytes(), before)

    def test_broken_erkannt(self):
        self._generate()
        (self.d / "c2pa_chain.pem").write_bytes(b"garbage")
        self.assertEqual(self.mc.bundle_state(), "broken")
        sys.argv = ["make_c2pa_cert.py"]
        self.assertEqual(self.mc.main(), 1)


if __name__ == "__main__":
    unittest.main()
