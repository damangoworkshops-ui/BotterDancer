"""Preflight-Gates + Postconditions von submit_workflow (Audit 04.08. F2/F7/F9
und Artefaktvertrag). ffmpeg-abhaengige Tests werden ohne ffmpeg uebersprungen."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import submit_workflow as sw  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or sw.FFPROBE_FALLBACK.replace("ffprobe.exe", "ffmpeg.exe")
HAVE_FFMPEG = os.path.isfile(FFMPEG) if not shutil.which("ffmpeg") else True


def make_video(path, fps, frames=8):
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=64x64:r=%d" % fps, "-frames:v", str(frames),
                    "-pix_fmt", "yuv420p", "-y", path], check=True, capture_output=True)


class TestCountImages(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        for i in range(5):
            open(os.path.join(self.d, "pose_%04d.png" % i), "wb").close()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_zaehlung(self):
        self.assertEqual(sw.count_usable_images(self.d, 0, 1), 5)
        self.assertEqual(sw.count_usable_images(self.d, 1, 1), 4)
        self.assertEqual(sw.count_usable_images(self.d, 0, 2), 3)


class TestSurplusGate(unittest.TestCase):
    """Audit F2: Ueberschuss-Verzeichnis (unresampeltes Material) muss FEHLER sein."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        for i in range(10):
            open(os.path.join(self.d, "pose_%04d.png" % i), "wb").close()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_surplus_ist_fehler(self):
        prompt = {"20": {"class_type": "VHS_LoadImagesPath", "inputs": {
            "directory": self.d, "skip_first_images": 0,
            "select_every_nth": 1, "image_load_cap": 3}}}
        issues = sw.preflight(prompt, force=True)  # force: nicht sys.exit, Liste pruefen
        fehler = [t for lv, t in issues if lv == "FEHLER"]
        self.assertTrue(any("MEHR" in t for t in fehler), issues)

    def test_passend_ist_ok(self):
        prompt = {"20": {"class_type": "VHS_LoadImagesPath", "inputs": {
            "directory": self.d, "skip_first_images": 0,
            "select_every_nth": 1, "image_load_cap": 10}}}
        issues = sw.preflight(prompt, force=True)
        self.assertFalse([t for lv, t in issues if lv == "FEHLER"], issues)


class TestDraftAngleGate(unittest.TestCase):
    """Audit F7: Draft-Presets >= 120 Grad muessen hart abgelehnt werden."""

    def _preset(self, angle, profile):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"workflow": "wan_animate_draft.json", "angle": angle,
                   "profile": profile}, f)
        f.close()
        return f.name

    def test_draft_120_gesperrt(self):
        p = self._preset(120, "draft")
        try:
            with self.assertRaises(SystemExit) as cm:
                sw.load_preset(p)
            self.assertEqual(cm.exception.code, 2)
        finally:
            os.unlink(p)

    def test_draft_120_mit_force_erlaubt(self):
        p = self._preset(120, "draft")
        try:
            wf, sets, desc = sw.load_preset(p, force=True)
            self.assertTrue(wf.endswith("wan_animate_draft.json"))
        finally:
            os.unlink(p)

    def test_draft_90_und_final_180_erlaubt(self):
        for angle, profile in ((90, "draft"), (180, "final")):
            p = self._preset(angle, profile)
            try:
                sw.load_preset(p)
            finally:
                os.unlink(p)


class TestPosePostcondition(unittest.TestCase):
    """Audit F2: SavePoseKps meldet keine Dateien an die History — das Tool
    muss die JSON selbst finden und die Framezahl pruefen."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self._old = sw.COMFY_OUTPUT
        sw.COMFY_OUTPUT = self.d
        self.prompt = {
            "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "x.mp4", "frame_load_cap": 5}},
            "2": {"class_type": "SavePoseKpsAsJsonFile", "inputs": {"filename_prefix": "tp"}},
        }

    def tearDown(self):
        sw.COMFY_OUTPUT = self._old
        shutil.rmtree(self.d, ignore_errors=True)

    def _write(self, n_frames):
        with open(os.path.join(self.d, "tp_00001.json"), "w") as f:
            json.dump([{"people": []}] * n_frames, f)

    def test_datei_fehlt(self):
        problems = sw.check_pose_postcondition(self.prompt, [], time.time() - 5)
        self.assertTrue(any("KEINE neue Datei" in p for p in problems))

    def test_verkuerzt(self):
        self._write(3)
        problems = sw.check_pose_postcondition(self.prompt, [], time.time() - 5)
        self.assertTrue(any("3 Frame-Eintraege" in p for p in problems))

    def test_cap_minus_1_toleriert(self):
        self._write(4)  # cap=5, Treiber liefert oft cap-1 (validierte Toleranz)
        files = []
        problems = sw.check_pose_postcondition(self.prompt, files, time.time() - 5)
        self.assertEqual(problems, [])
        self.assertEqual(len(files), 1)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg nicht verfuegbar")
class TestRifeFpsGate(unittest.TestCase):
    """Audit F9: RIFE-Kette muss input_fps * multiplier == frame_rate erzwingen."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.v16 = os.path.join(cls.d, "in16.mp4")
        cls.v32 = os.path.join(cls.d, "in32.mp4")
        make_video(cls.v16, 16)
        make_video(cls.v32, 32)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def _graph(self, video):
        return {
            "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": video, "force_rate": 0}},
            "2": {"class_type": "RIFE VFI", "inputs": {"multiplier": 2}},
            "3": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 32,
                                                               "save_metadata": False}},
        }

    def test_16fps_ok(self):
        issues = []
        sw.check_rife_fps(self._graph(self.v16), issues)
        self.assertTrue(any(lv == "OK" for lv, _ in issues), issues)

    def test_32fps_fehler(self):
        issues = []
        sw.check_rife_fps(self._graph(self.v32), issues)
        self.assertTrue(any(lv == "FEHLER" for lv, _ in issues), issues)

    def test_force_rate_macht_invariant(self):
        g = self._graph(self.v32)
        g["1"]["inputs"]["force_rate"] = 16
        issues = []
        sw.check_rife_fps(g, issues)
        self.assertEqual(issues, [])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg nicht verfuegbar")
class TestVideoArtefaktvertrag(unittest.TestCase):
    """Artefaktvertrag: Output-fps und -Framezahl muessen dem Graph entsprechen."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.v16 = os.path.join(cls.d, "out16.mp4")
        make_video(cls.v16, 16, frames=8)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def _graph(self, fps, length):
        return {
            "1": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": fps}},
            "2": {"class_type": "WanAnimateToVideo", "inputs": {"length": length}},
        }

    def test_konsistent(self):
        self.assertEqual(
            sw.check_video_postconditions(self._graph(16, 8), [self.v16]), [])

    def test_falsche_fps(self):
        problems = sw.check_video_postconditions(self._graph(32, 8), [self.v16])
        self.assertTrue(any("fps" in p for p in problems), problems)

    def test_falsche_framezahl(self):
        problems = sw.check_video_postconditions(self._graph(16, 81), [self.v16])
        self.assertTrue(any("81" in p for p in problems), problems)

    def test_rife_frame_semantik(self):
        """RIFE: N Input-Frames -> N*mult-(mult-1) Output (interpoliert nur
        ZWISCHEN Frames; 81->161 im validierten Juli-Lauf, live bestaetigt 04.08.)."""
        graph = {
            "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": self.v16,
                                                            "force_rate": 0}},
            "2": {"class_type": "RIFE VFI", "inputs": {"multiplier": 2}},
            "3": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 32}},
        }
        fps, frames = sw.expected_video_contract(graph)
        self.assertEqual(fps, 32)
        self.assertEqual(frames, 8 * 2 - 1)


if __name__ == "__main__":
    unittest.main()
