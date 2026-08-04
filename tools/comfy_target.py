"""BotterDancer — gemeinsame Definition der PROJEKT-ComfyUI-Instanz (Audit 04.08., F1).

Hintergrund: Seit dem Reboot am 03.08. belegt die ComfyUI-DESKTOP-App (v0.30,
eigene Model-/Input-/Output-Pfade unter Documents\\ComfyUI) bevorzugt Port 8188.
Alle Tools, die blind auf 8188 zeigten, trafen damit die falsche Instanz —
Submits, /free, /interrupt und History gingen an einen Server mit fremden Pfaden.

Dieses Modul ist die EINZIGE Quelle fuer "wo ist unsere Instanz":
- fingerprint(url): identifiziert die Projekt-Instanz an comfyui_version 0.22.x
  UND Abwesenheit von --base-directory in argv (Desktop-App setzt das immer).
- find_project_server(): probiert die Kandidaten-Ports und liefert die erste
  URL mit Projekt-Fingerprint — oder None plus Diagnose, was stattdessen lauscht.

Aendert sich die Projekt-Installation (Upgrade von v0.22), muss
EXPECT_VERSION_PREFIX hier nachgezogen werden — der Fehler ist dann laut.
"""
import json
import urllib.request

CANDIDATE_URLS = ["http://127.0.0.1:8188", "http://127.0.0.1:8190"]
EXPECT_VERSION_PREFIX = "0.22"
COMFY_ROOT = r"C:\ComfyUI"


def fingerprint(url, timeout=4):
    """(is_project, info_string). Wirft bei Nichterreichbarkeit (URLError etc.)."""
    with urllib.request.urlopen(url + "/system_stats", timeout=timeout) as r:
        stats = json.load(r)
    system = stats.get("system", {})
    version = str(system.get("comfyui_version", "?"))
    argv = " ".join(str(a) for a in system.get("argv", []))
    is_project = version.startswith(EXPECT_VERSION_PREFIX) and "--base-directory" not in argv
    label = "PROJEKT" if is_project else "FREMD"
    return is_project, f"{label}: v{version}" + (" (Desktop-App)" if "--base-directory" in argv else "")


def find_project_server(preferred=None, timeout=4):
    """(url|None, [(url, diagnose), ...] fuer alle probierten Kandidaten)."""
    tried = []
    urls = ([preferred] if preferred else []) + [u for u in CANDIDATE_URLS if u != preferred]
    for u in urls:
        try:
            ok, info = fingerprint(u, timeout=timeout)
        except Exception as e:
            tried.append((u, f"nicht erreichbar ({e.__class__.__name__})"))
            continue
        if ok:
            return u, tried
        tried.append((u, info))
    return None, tried
