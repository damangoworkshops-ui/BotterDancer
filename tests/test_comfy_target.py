"""Fingerprint + Auto-Discovery der Projekt-Instanz (Audit 04.08. F1).
Rein lokal: gefakte /system_stats via In-Process-HTTP-Server."""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import comfy_target  # noqa: E402

PROJECT_STATS = {"system": {"comfyui_version": "0.22.0", "argv": [
    "main.py", "--listen", "127.0.0.1", "--port", "8190", "--reserve-vram", "4"]}}
DESKTOP_STATS = {"system": {"comfyui_version": "0.30.0", "argv": [
    "ComfyUI\\main.py", "--base-directory", "C:\\Users\\x\\Documents\\ComfyUI"]}}


def serve(payload):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_port


class TestFingerprint(unittest.TestCase):
    def test_project_erkannt(self):
        srv, url = serve(PROJECT_STATS)
        try:
            ok, info = comfy_target.fingerprint(url)
            self.assertTrue(ok)
            self.assertIn("PROJEKT", info)
        finally:
            srv.shutdown()

    def test_desktop_ist_fremd(self):
        srv, url = serve(DESKTOP_STATS)
        try:
            ok, info = comfy_target.fingerprint(url)
            self.assertFalse(ok)
            self.assertIn("Desktop-App", info)
        finally:
            srv.shutdown()

    def test_falsche_version_ist_fremd(self):
        srv, url = serve({"system": {"comfyui_version": "0.30.0", "argv": ["main.py"]}})
        try:
            ok, _ = comfy_target.fingerprint(url)
            self.assertFalse(ok)
        finally:
            srv.shutdown()


class TestDiscovery(unittest.TestCase):
    def test_ueberspringt_desktop_findet_projekt(self):
        s1, desktop_url = serve(DESKTOP_STATS)
        s2, project_url = serve(PROJECT_STATS)
        old = comfy_target.CANDIDATE_URLS
        comfy_target.CANDIDATE_URLS = [desktop_url, project_url]
        try:
            url, tried = comfy_target.find_project_server()
            self.assertEqual(url, project_url)
            self.assertEqual(len(tried), 1)
            self.assertIn("FREMD", tried[0][1])
        finally:
            comfy_target.CANDIDATE_URLS = old
            s1.shutdown()
            s2.shutdown()

    def test_nichts_gefunden(self):
        old = comfy_target.CANDIDATE_URLS
        comfy_target.CANDIDATE_URLS = ["http://127.0.0.1:1"]  # nie erreichbar
        try:
            url, tried = comfy_target.find_project_server(timeout=1)
            self.assertIsNone(url)
            self.assertIn("nicht erreichbar", tried[0][1])
        finally:
            comfy_target.CANDIDATE_URLS = old


if __name__ == "__main__":
    unittest.main()
