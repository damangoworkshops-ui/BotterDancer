"""Integrationstests fuer den C2PA-Export (v2 nach Doppel-Audit 04.08.):
Happy Path mit frischem Wegwerf-Schluesselmaterial, Eingabevertrag, Payload-Gates.
Braucht c2pa-python, imwatermark und ffmpeg — sonst Skip. Kein Zugriff auf
assets/keys/ oder persoenliche Medien; Video-Fixtures sind synthetisch."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

try:
    import c2pa  # noqa: F401
    from imwatermark import WatermarkEncoder  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

import export_clip as ec  # noqa: E402  (Modul-Import ist leichtgewichtig)

FFMPEG = shutil.which("ffmpeg") or (ec.FFMPEG_FALLBACKS[0] + r"\ffmpeg.exe")
HAVE_FFMPEG = os.path.isfile(FFMPEG) if not shutil.which("ffmpeg") else True


def make_video(path, with_audio=False, frames=8, size="512x768"):
    # Fixture-Wahl ist empirisch (04.08.): dwtDct bettet NUR im U-Chroma-Kanal
    # ein — testsrc2/mandelbrot haben flaechige Chromas, dort ueberlebt das
    # Watermark keine verlustbehaftete Kompression (gleicher Mechanismus wie
    # Codex-H8-Weissframe-Befund). Chroma-Rauschen traegt zuverlaessig, wie
    # der echte Fell-Content der Pipeline. Input hochwertig (crf 8) encodieren,
    # damit die Einbettung auf sauberem Material passiert.
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i",
           "color=c=0x808080:s=%s:r=16,noise=alls=40:allf=t" % size]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    cmd += ["-frames:v", str(frames), "-pix_fmt", "yuv420p", "-crf", "8"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-y", path], check=True, capture_output=True)


@unittest.skipUnless(HAVE_DEPS and HAVE_FFMPEG, "c2pa/imwatermark/ffmpeg nicht verfuegbar")
class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import make_c2pa_cert as mc
        cls.d = Path(tempfile.mkdtemp())
        cls.keys = cls.d / "keys"
        # Wegwerf-Schluessel NUR fuer den Test
        mc.KEYS_DIR = cls.keys
        argv, sys.argv = sys.argv, ["make_c2pa_cert.py"]
        try:
            assert mc.main() == 0
        finally:
            sys.argv = argv
        cls._old_keys = ec.KEYS_DIR
        ec.KEYS_DIR = cls.keys
        cls.video = str(cls.d / "in.mp4")
        make_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        ec.KEYS_DIR = cls._old_keys
        shutil.rmtree(cls.d, ignore_errors=True)

    def _run(self, *argv):
        old = sys.argv
        sys.argv = ["export_clip.py", *argv]
        try:
            return ec.main()
        finally:
            sys.argv = old

    def test_happy_path_mit_watermark(self):
        outdir = str(self.d / "export1")
        self.assertEqual(self._run(self.video, "--outdir", outdir), 0)
        out = Path(outdir) / "in_export.mp4"
        self.assertTrue(out.is_file())
        # C2PA-Manifest im Ergebnis lesbar + created-Assertion (Audit-Regression)
        import json as _json
        from c2pa import Reader
        with Reader(str(out)) as r:
            detailed = _json.loads(r.detailed_json())
        claim = detailed["manifests"][detailed["active_manifest"]]["claim"]
        created = " ".join(a.get("url", "") for a in claim.get("created_assertions", []))
        self.assertIn("c2pa.actions", created)

    def test_audio_verletzt_vertrag(self):
        va = str(self.d / "audio.mp4")
        make_video(va, with_audio=True)
        with self.assertRaises(SystemExit) as cm:
            self._run(va, "--outdir", str(self.d / "export2"))
        self.assertEqual(cm.exception.code, 2)

    def test_payload_gates(self):
        for bad in (["--wm-payload", "ZULANGEPAYLOAD"], ["--crf", "50"]):
            with self.assertRaises(SystemExit) as cm:
                self._run(self.video, "--outdir", str(self.d / "export3"), *bad)
            self.assertEqual(cm.exception.code, 2)

    def test_audio_mux(self):
        """Audio-Lane: stummes Video + --audio wav -> Export mit genau 1 Video-
        und 1 Audio-Stream, Verify inkl. Stream-Inventar gruen."""
        wav = str(self.d / "beat.wav")
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=220:duration=2", "-y", wav],
                       check=True, capture_output=True)
        outdir = str(self.d / "export6")
        self.assertEqual(self._run(self.video, "--outdir", outdir, "--audio", wav), 0)
        ffprobe = ec.find_tool("ffprobe")
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type",
                            "-of", "csv=p=0", str(Path(outdir) / "in_export.mp4")],
                           capture_output=True, text=True)
        self.assertEqual(sorted(r.stdout.split()), ["audio", "video"])

    def test_zu_klein_fuers_watermark(self):
        """Regression: imwatermark verlangt >=256x256 — muss sauberer
        Vertragsfehler sein, kein RuntimeError tief im Encode."""
        small = str(self.d / "small.mp4")
        make_video(small, size="64x64")
        with self.assertRaises(SystemExit) as cm:
            self._run(small, "--outdir", str(self.d / "export5"))
        self.assertEqual(cm.exception.code, 2)

    def test_kollision_ohne_overwrite(self):
        outdir = str(self.d / "export4")
        self.assertEqual(self._run(self.video, "--outdir", outdir, "--no-watermark"), 0)
        with self.assertRaises(SystemExit) as cm:
            self._run(self.video, "--outdir", outdir, "--no-watermark")
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
