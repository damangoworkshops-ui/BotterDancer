r"""BotterDancer — Workflow-Submit mit Preflight, Poll und lauten Fehlern.

Schliesst die Kernluecke aus AUDIT-2026-07-18/19 (A3/P1-7): bisher war jeder
Render ein blinder POST /prompt ohne Fehler-Feedback, plus "neueste Datei aus
dem Output-Ordner fischen". Dieses Tool:

- laedt ein workflows\*.json (Format: {"prompt": {<node-map>}} oder roher Node-Map),
- patcht Inputs per --set NODE.INPUT=WERT (z.B. --set 1.video=botter_final_00002.mp4
  gegen die hartkodierte RIFE-Falle, --set 31.seed=123),
- Preflight VOR dem POST: Server erreichbar, referenzierte Inputs existieren,
  PNG-Anzahl im Pose-Verzeichnis passt zu image_load_cap/length (der
  Konsistenz-Check aus dem Audit-Hauptbefund),
- POST /prompt BOM-frei (Python, nicht PS 5.1 — die BOM-Fehlklasse mit 7
  belegten Fehl-Submissions entfaellt damit strukturell),
- pollt /history/<id> bis Erfolg/Fehler, zeigt tqdm-Fortschritt aus dem
  UTF-8-Log, druckt Node-Fehler im Klartext,
- druckt am Ende die ECHTEN Output-Dateipfade aus der History (kein Fischen).

Presets (loesen die 9-Kopien-Drift, Audit P1-9): workflows\presets\<name>.json =
  {"workflow": "wan_animate_final_v2.json", "set": ["20.directory=...", "33.filename_prefix=..."]}
Der Graph existiert nur noch EINMAL; Presets patchen ihn. CLI---set gewinnt ueber Preset-set.

Nutzung (jedes Python 3.8+, stdlib-only):
  python tools\submit_workflow.py workflows\wan_animate_final_v2.json
  python tools\submit_workflow.py --preset cam180
  python tools\submit_workflow.py workflows\rife_32fps.json --set 1.video=botter_final_00002.mp4
  python tools\submit_workflow.py --preset cam90 --check          (nur Preflight)
  python tools\submit_workflow.py workflows\rife_32fps.json --set 1.video=x.mp4 --dump

Exit-Codes: 0 Erfolg | 1 Job-Fehler (Node-Exception) | 2 Preflight/Usage |
            3 Server nicht erreichbar | 4 Submit abgelehnt | 5 Timeout
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from fractions import Fraction

import comfy_target  # gemeinsame Projekt-Instanz-Definition (Audit 04.08., F1)

COMFY_URL = "http://127.0.0.1:8188"
FFPROBE_FALLBACK = (r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages"
                    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                    r"\ffmpeg-8.1-full_build\bin\ffprobe.exe")
COMFY_ROOT = r"C:\ComfyUI"
COMFY_INPUT = r"C:\ComfyUI\input"
COMFY_OUTPUT = r"C:\ComfyUI\output"
COMFY_TEMP = r"C:\ComfyUI\temp"
COMFY_LOG = r"C:\ComfyUI\comfyui_utf8.log"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?([\d.]+)\s*(s/it|it/s)\]")
ANNOT_RE = re.compile(r"^(.*) \[(input|output|temp)\]$")


def resolve_annotated(name):
    """ComfyUI-Annotationssyntax 'datei.png [input|output|temp]' aufloesen —
    VOR dem isabs-Check, wie folder_paths.get_annotated_filepath es tut."""
    base = COMFY_INPUT
    m = ANNOT_RE.match(name)
    if m:
        name = m.group(1)
        base = {"input": COMFY_INPUT, "output": COMFY_OUTPUT, "temp": COMFY_TEMP}[m.group(2)]
    return name if os.path.isabs(name) else os.path.join(base, name)


def _lit(v):
    """Literal-Zahl oder None, wenn der Input verkabelt ist (Node-Link ['id', slot])."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def http_json(url, payload=None, timeout=6):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")  # ohne BOM, immer UTF-8
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}


def load_workflow(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig: schluckt evtl. BOM
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"FEHLER: Workflow {path}: {e}", file=sys.stderr)
        sys.exit(2)
    prompt = doc.get("prompt", doc)
    if not isinstance(prompt, dict) or not all(
            isinstance(v, dict) and "class_type" in v for v in prompt.values()):
        print(f"FEHLER: {path} ist kein API-Format-Workflow (Node-Map mit class_type).",
              file=sys.stderr)
        sys.exit(2)
    return prompt


def apply_sets(prompt, sets):
    for s in sets:
        if "=" not in s or "." not in s.split("=", 1)[0]:
            print(f"FEHLER: --set erwartet NODE.INPUT=WERT, bekommen: {s}", file=sys.stderr)
            sys.exit(2)
        key, raw = s.split("=", 1)
        node_id, input_name = key.split(".", 1)
        if node_id not in prompt:
            print(f"FEHLER: --set {s}: Node {node_id} existiert nicht im Workflow "
                  f"(vorhanden: {', '.join(sorted(prompt, key=lambda x: int(x) if x.isdigit() else 0))})",
                  file=sys.stderr)
            sys.exit(2)
        try:
            value = json.loads(raw)  # Zahlen/bool/null/Strings-in-Quotes
        except ValueError:
            value = raw              # roher String (auch Windows-Pfade)
        prompt[node_id].setdefault("inputs", {})
        if input_name not in prompt[node_id]["inputs"]:
            print(f"WARNUNG: Node {node_id} ({prompt[node_id].get('class_type')}) hatte "
                  f"bisher keinen Input '{input_name}' — wird neu gesetzt.")
        prompt[node_id]["inputs"][input_name] = value
        print(f"  set {node_id}.{input_name} = {value!r}")
    return prompt


def count_usable_images(directory, skip_first, every_nth):
    try:
        names = [n for n in os.listdir(directory) if n.lower().endswith(IMG_EXTS)]
    except OSError:
        return None
    n = max(0, len(names) - max(0, skip_first))
    if every_nth and every_nth > 1:
        n = (n + every_nth - 1) // every_nth
    return n


def ffprobe_video(path):
    """{'fps': float, 'frames': int|None} oder None (ffprobe fehlt/Datei unlesbar)."""
    import shutil as _sh
    exe = _sh.which("ffprobe") or (FFPROBE_FALLBACK if os.path.isfile(FFPROBE_FALLBACK) else None)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=avg_frame_rate,nb_frames", "-of", "json", path],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        s = json.loads(r.stdout)["streams"][0]
        frames = s.get("nb_frames")
        return {"fps": float(Fraction(s["avg_frame_rate"])),
                "frames": int(frames) if frames and str(frames).isdigit() else None}
    except (ValueError, ZeroDivisionError, OSError, KeyError, IndexError,
            subprocess.TimeoutExpired):
        return None


def ffprobe_fps(path):
    info = ffprobe_video(path)
    return info["fps"] if info else None


def check_rife_fps(prompt, issues):
    """Audit 04.08. F9: RIFE-Graph kodiert die Output-fps hart (VHS_VideoCombine),
    prueft aber die Eingangs-fps nie — ein 32-fps-Input ergaebe Zeitlupe mit ERFOLG.
    Erwartung wird aus dem Graph abgeleitet: input_fps * multiplier == frame_rate."""
    mult = rate = video = None
    for n in prompt.values():
        ct = n.get("class_type", "")
        ins = n.get("inputs", {})
        if "RIFE" in ct:
            mult = _lit(ins.get("multiplier"))
        elif ct == "VHS_VideoCombine":
            rate = _lit(ins.get("frame_rate"))
        elif ct == "VHS_LoadVideo":
            if _lit(ins.get("force_rate")) not in (0, None):
                return  # force_rate resampelt -> Kette ist fps-invariant
            if isinstance(ins.get("video"), str):
                video = resolve_annotated(ins["video"])
    if not (mult and rate and video and os.path.isfile(video)):
        return
    fps = ffprobe_fps(video)
    if fps is None:
        issues.append(("WARNUNG", f"RIFE-Kette: Eingangs-fps nicht pruefbar (ffprobe fehlt?) — "
                       f"erwartet waeren {rate}/{mult}={rate / mult:g} fps."))
    elif abs(fps * mult - rate) > 0.05:
        issues.append(("FEHLER", f"RIFE-Kette: Input hat {fps:g} fps, Workflow erwartet "
                       f"{rate / mult:g} (Output ist hart {rate} fps bei x{mult}) — "
                       f"Ergebnis waere Zeitlupe/Laengung. Falschen Input gesetzt?"))
    else:
        issues.append(("OK", f"RIFE-Kette: {fps:g} fps x{mult} -> {rate} fps konsistent."))


def preflight(prompt, force=False):
    """Statische Checks gegen Dateisystem + Graph. Returns Liste (level, text).
    Verkabelte (Node-Link-)Inputs sind statisch nicht pruefbar -> WARNUNG statt
    TypeError (Re-Attack-Befund: GUI-Exporte mit Primitives crashten das Tool)."""
    issues = []
    lengths_raw = [n["inputs"].get("length") for n in prompt.values()
                   if n.get("class_type") == "WanAnimateToVideo"]
    lengths = [v for v in lengths_raw if _lit(v) is not None]
    if len(lengths) != len(lengths_raw):
        issues.append(("WARNUNG", "WanAnimateToVideo.length ist verkabelt — "
                       "Laengen-Konsistenz-Check uebersprungen."))
    length = lengths[0] if lengths else None

    for nid, node in sorted(prompt.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        ct = node.get("class_type", "?")
        ins = node.get("inputs", {})
        if ct == "VHS_LoadImagesPath":
            d = ins.get("directory", "")
            if isinstance(d, str) and d and not os.path.isabs(d):
                # ComfyUI loest relative Pfade serverseitig gegen C:\ComfyUI auf
                d = os.path.join(COMFY_ROOT, d)
                issues.append(("WARNUNG", f"Node {nid}: relatives directory — Aufloesung "
                               f"gegen {COMFY_ROOT} angenommen."))
            if not isinstance(d, str) or not os.path.isdir(d):
                issues.append(("FEHLER", f"Node {nid}: Verzeichnis fehlt: {d}"))
            else:
                skip = _lit(ins.get("skip_first_images", 0))
                nth = _lit(ins.get("select_every_nth", 1))
                cap = _lit(ins.get("image_load_cap", 0))
                avail = (count_usable_images(d, skip, nth)
                         if None not in (skip, nth, cap) else None)
                if None in (skip, nth, cap):
                    issues.append(("WARNUNG", f"Node {nid}: cap/skip/nth verkabelt — "
                                   f"Frame-Konsistenz-Check uebersprungen."))
                elif avail is None:
                    issues.append(("WARNUNG", f"Node {nid}: Verzeichnis nicht lesbar — "
                                   f"Frame-Konsistenz-Check uebersprungen."))
                else:
                    eff = min(avail, cap) if cap else avail
                    info = f"Node {nid}: {avail} Bilder in {os.path.basename(d)}, cap={cap} -> laedt {eff}"
                    surplus = bool(cap and avail > cap + 1)
                    if surplus:
                        # Audit 04.08. F2: war WARNUNG — der belegte 81/150-Zeitlupenfehler
                        # lief damit bis ERFOLG durch. Jetzt FEHLER; --force bleibt Override.
                        issues.append(("FEHLER", info + f" — Verzeichnis hat {avail - eff} Frames MEHR "
                                       f"als geladen werden. Unresampeltes 30fps-Material? Dann zeigt der "
                                       f"Render nur {eff}/{avail} der Choreo in Zeitlupe (Audit-Hauptbefund). "
                                       f"Fix: reproject_camera.py --target-fps 16 und Ordner regenerieren."))
                    if length is not None and eff < length - 1:
                        # Akzeptierte Konstellation: length == eff oder length == eff+1
                        # (WanAnimate polstert den letzten Latent-Frame; final_v2 laeuft validiert mit 80 PNGs/length 81).
                        issues.append(("FEHLER", info + f" — zu WENIG fuer length={length} "
                                       f"(min. {length - 1}). Pose-Verzeichnis regenerieren?"))
                    elif not surplus:
                        issues.append(("OK", info + (f", length={length}" if length is not None else "")))
        elif ct == "VHS_LoadVideo":
            v = ins.get("video", "")
            if not isinstance(v, str):
                issues.append(("WARNUNG", f"Node {nid}: video ist verkabelt — Check uebersprungen."))
            else:
                p = resolve_annotated(v)
                if not os.path.isfile(p):
                    issues.append(("FEHLER", f"Node {nid}: Video fehlt: {p}"))
                else:
                    mtime = time.strftime("%d.%m. %H:%M", time.localtime(os.path.getmtime(p)))
                    issues.append(("OK", f"Node {nid}: {v} ({os.path.getsize(p) / 1e6:.1f} MB, vom {mtime})"))
        elif ct == "LoadImage":
            img = ins.get("image", "")
            if isinstance(img, str):
                p = resolve_annotated(img)
                if not os.path.isfile(p):
                    issues.append(("FEHLER", f"Node {nid}: Bild fehlt: {p}"))
        elif ct == "VHS_VideoCombine":
            if ins.get("save_metadata") is not False:
                issues.append(("WARNUNG", f"Node {nid}: save_metadata nicht false — "
                               f"MP4 leakt den kompletten Workflow-Graph (Audit P0-2)."))
        # Verkabelung: referenzierte Quell-Nodes muessen existieren
        for iname, val in ins.items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in prompt:
                    issues.append(("FEHLER", f"Node {nid}.{iname} verweist auf fehlenden Node {val[0]}"))

    check_rife_fps(prompt, issues)

    for level, text in issues:
        print(f"  [{level}] {text}")
    errors = [t for lv, t in issues if lv == "FEHLER"]
    if errors and not force:
        print(f"\nPreflight FEHLGESCHLAGEN ({len(errors)} Fehler). --force uebergeht das.",
              file=sys.stderr)
        sys.exit(2)
    return issues


def tail_progress(min_pos=0):
    """Letzte tqdm-Zeile aus dem UTF-8-Log (Anzeige-only, best effort).
    min_pos: Log-Groesse zum Submit-Zeitpunkt — aeltere Zeilen (voriger Job,
    'Step 20/20') werden nicht als Fortschritt des NEUEN Jobs angezeigt."""
    try:
        size = os.path.getsize(COMFY_LOG)
        if min_pos > size:  # Log wurde rotiert/archiviert
            min_pos = 0
        with open(COMFY_LOG, "rb") as f:
            f.seek(max(0, size - 32768, min_pos))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    last = None
    for line in re.split(r"[\r\n]+", chunk):
        m = TQDM_RE.search(line)
        if m:
            last = m
    if not last:
        return None
    cur, total, rate, unit = int(last.group(1)), int(last.group(2)), float(last.group(3)), last.group(4)
    s_it = rate if unit == "s/it" else (1.0 / rate if rate else 0.0)
    eta = (total - cur) * s_it
    return f"Step {cur}/{total} @ {s_it:.1f}s/it, ETA {eta:.0f}s"


def collect_outputs(hist_entry):
    files = []
    for node_out in hist_entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for f in node_out.get(key, []):
                if not f.get("filename"):
                    continue
                base = {"output": COMFY_OUTPUT, "input": COMFY_INPUT,
                        "temp": COMFY_TEMP}.get(f.get("type", "output"), COMFY_OUTPUT)
                files.append(os.path.join(base, f.get("subfolder", ""), f["filename"]))
    return files


def print_job_error(status):
    for msg in status.get("messages", []):
        if msg[0] == "execution_error":
            d = msg[1]
            print(f"\nJOB-FEHLER in Node {d.get('node_id')} ({d.get('node_type')}):",
                  file=sys.stderr)
            print("  " + (d.get("exception_message") or "?"), file=sys.stderr)
            for line in (d.get("traceback") or [])[-4:]:
                print("  | " + line.rstrip(), file=sys.stderr)
        elif msg[0] == "execution_interrupted":
            print("\nJOB UNTERBROCHEN (interrupt).", file=sys.stderr)


def load_preset(name, force=False):
    """Preset aufloesen: Name -> workflows\\presets\\<name>.json, oder direkter Pfad."""
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = name if name.lower().endswith(".json") else os.path.join(
        project, "workflows", "presets", name + ".json")
    if not os.path.isfile(p):
        preset_dir = os.path.join(project, "workflows", "presets")
        avail = sorted(n[:-5] for n in os.listdir(preset_dir)) if os.path.isdir(preset_dir) else []
        print(f"FEHLER: Preset nicht gefunden: {p}"
              + (f"\n  Verfuegbar: {', '.join(avail)}" if avail else ""), file=sys.stderr)
        sys.exit(2)
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            preset = json.load(f)
    except (OSError, ValueError) as e:
        print(f"FEHLER: Preset {p}: {e}", file=sys.stderr)
        sys.exit(2)
    wf = preset.get("workflow")
    if not wf:
        print(f"FEHLER: Preset {p} hat kein 'workflow'-Feld.", file=sys.stderr)
        sys.exit(2)
    # Audit 04.08. F7: Draft-Lane ist fuer Rueckwinkel gesperrt (GO-CHECKLIST 03.08.:
    # cam120-Draft kollabiert in Sitzpose; Regel: Draft nur 0-90 Grad).
    angle = preset.get("angle")
    if (preset.get("profile") == "draft" and isinstance(angle, (int, float)) and angle >= 120):
        msg = (f"Preset {os.path.basename(p)}: Draft-Lane fuer Winkel >= 120 Grad gesperrt "
               f"(angle={angle:g}; cam120-Draft-Kollaps 03.08.). Full-Quality-Preset nutzen.")
        if force:
            print(f"  [WARNUNG] {msg} (--force uebergeht das)")
        else:
            print(f"FEHLER: {msg} --force uebergeht das bewusst.", file=sys.stderr)
            sys.exit(2)
    if not os.path.isabs(wf):
        wf = os.path.join(project, "workflows", wf)
    return wf, list(preset.get("set", [])), preset.get("description", "")


def check_pose_postcondition(prompt, files, t_submit):
    """Audit 04.08. F2: SavePoseKpsAsJsonFile meldet seine Datei NICHT an die
    History (RETURN_TYPES=(), return {}) — Erfolg hiess bisher auch dann
    'ERFOLG (keine Dateien gemeldet)', wenn keine oder eine verkuerzte JSON
    entstand. Sucht die Datei selbst und prueft Frames gegen frame_load_cap."""
    prefixes = [n["inputs"].get("filename_prefix") for n in prompt.values()
                if n.get("class_type") == "SavePoseKpsAsJsonFile"]
    prefixes = [p for p in prefixes if isinstance(p, str) and p]
    if not prefixes:
        return []
    caps = [_lit(n["inputs"].get("frame_load_cap")) for n in prompt.values()
            if n.get("class_type") == "VHS_LoadVideo"]
    cap = next((c for c in caps if c), None)
    problems = []
    for prefix in prefixes:
        pattern = os.path.join(COMFY_OUTPUT, prefix + "_*.json")
        cands = [p for p in glob.glob(pattern) if os.path.getmtime(p) >= t_submit - 2]
        if not cands:
            problems.append(f"SavePoseKps: KEINE neue Datei {pattern} seit Submit entstanden.")
            continue
        newest_p = max(cands, key=os.path.getmtime)
        try:
            with open(newest_p, "r", encoding="utf-8") as f:
                n_frames = len(json.load(f))
        except (OSError, ValueError) as e:
            problems.append(f"SavePoseKps: {newest_p} nicht lesbar: {e}")
            continue
        if cap and not (cap - 1 <= n_frames <= cap):
            problems.append(f"SavePoseKps: {os.path.basename(newest_p)} hat {n_frames} "
                            f"Frame-Eintraege, erwartet {cap - 1}-{cap} (frame_load_cap={cap}) "
                            f"— verkuerzter/falscher Driver?")
        else:
            files.append(f"{newest_p}  ({n_frames} Frame-Eintraege, verifiziert)")
    return problems


def expected_video_contract(prompt):
    """Erwartungen an Video-Outputs aus dem Graph ableiten (Artefaktvertrag,
    Audit 04.08. Folge-Item): fps aus VHS_VideoCombine.frame_rate; Framezahl aus
    WanAnimateToVideo.length bzw. RIFE: Input-Frames x multiplier."""
    fps = frames = None
    mult = video_in = None
    for n in prompt.values():
        ct = n.get("class_type", "")
        ins = n.get("inputs", {})
        if ct == "VHS_VideoCombine":
            fps = _lit(ins.get("frame_rate"))
        elif ct == "WanAnimateToVideo":
            frames = _lit(ins.get("length"))
        elif "RIFE" in ct:
            mult = _lit(ins.get("multiplier"))
        elif ct == "VHS_LoadVideo" and isinstance(ins.get("video"), str):
            video_in = resolve_annotated(ins["video"])
    if frames is None and mult and video_in and os.path.isfile(video_in):
        info = ffprobe_video(video_in)
        if info and info["frames"]:
            # RIFE interpoliert ZWISCHEN Frames, extrapoliert nicht ueber den
            # letzten hinaus: N Input -> N*mult-(mult-1) Output (81 -> 161 bei x2;
            # deckt sich mit dem validierten Juli-Lauf). Naives N*mult liess den
            # Vertrag am korrekten Output scheitern (04.08. live erwischt).
            frames = int(info["frames"] * mult - (mult - 1))
    return fps, frames


def check_video_postconditions(prompt, files):
    """Fertige Video-Outputs gegen den Graph-Vertrag pruefen. Rueckgabe: Problemliste."""
    exp_fps, exp_frames = expected_video_contract(prompt)
    problems = []
    for f in files:
        path = f.split("  (")[0]  # evtl. annotierte Eintraege (Pose-Postcondition)
        if not path.lower().endswith((".mp4", ".webm", ".mov")):
            continue
        if not os.path.isfile(path):
            problems.append(f"Output fehlt auf Platte: {path}")
            continue
        info = ffprobe_video(path)
        if info is None:
            problems.append(f"Output nicht analysierbar (ffprobe): {path}")
            continue
        if not info["frames"]:
            problems.append(f"{os.path.basename(path)}: 0/unbekannte Frames.")
            continue
        if exp_fps is not None and abs(info["fps"] - exp_fps) > 0.05:
            problems.append(f"{os.path.basename(path)}: {info['fps']:g} fps, Graph "
                            f"verlangt {exp_fps:g}.")
        if exp_frames is not None and info["frames"] != exp_frames:
            problems.append(f"{os.path.basename(path)}: {info['frames']} Frames, "
                            f"Graph erwartet {exp_frames}.")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow", nargs="?", help="Pfad zum Workflow-JSON (API-Format)")
    ap.add_argument("--preset", help="Preset-Name aus workflows\\presets\\ (oder Pfad)")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=WERT",
                    help="Input patchen, mehrfach erlaubt (gewinnt ueber Preset)")
    ap.add_argument("--server", default=None,
                    help="URL der Projekt-Instanz (Default: Auto-Discovery mit Fingerprint)")
    ap.add_argument("--check", action="store_true", help="nur Preflight, kein Submit")
    ap.add_argument("--dump", action="store_true", help="gepatchtes JSON ausgeben, kein Submit")
    ap.add_argument("--force", action="store_true", help="Preflight-Fehler ignorieren")
    ap.add_argument("--no-wait", action="store_true", help="nach Submit nicht pollen")
    ap.add_argument("--timeout", type=int, default=3600, help="Poll-Timeout in Sekunden")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # Windows-Konsole (cp1252) absichern
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    sets = []
    if args.preset:
        wf, preset_sets, desc = load_preset(args.preset, force=args.force)
        if args.workflow:
            print("FEHLER: entweder Workflow-Pfad ODER --preset, nicht beides.", file=sys.stderr)
            sys.exit(2)
        args.workflow = wf
        sets.extend(preset_sets)
        print(f"Preset {args.preset}: {desc or os.path.basename(wf)}")
    elif not args.workflow:
        print("FEHLER: Workflow-Pfad oder --preset angeben.", file=sys.stderr)
        sys.exit(2)
    sets.extend(args.set)  # CLI zuletzt -> gewinnt

    prompt = load_workflow(args.workflow)
    if sets:
        print("Overrides:")
        prompt = apply_sets(prompt, sets)
    if args.dump:
        json.dump({"prompt": prompt}, sys.stdout, indent=1)
        print()
        return

    print(f"Preflight: {os.path.basename(args.workflow)} ({len(prompt)} Nodes)")
    preflight(prompt, force=args.force)

    # Server-Identitaet (Audit 04.08. F1): nie blind gegen Port 8188 arbeiten —
    # dort kann die Desktop-App (v0.30, fremde Model-/Input-/Output-Pfade) lauschen.
    server, tried = comfy_target.find_project_server(preferred=args.server)
    if args.check:
        if server:
            print(f"Preflight OK (--check). Projekt-Instanz gefunden: {server}")
        else:
            diag = "; ".join(f"{u}: {d}" for u, d in tried)
            print(f"Preflight OK (Dateichecks; Projekt-Instanz NICHT gefunden — {diag})")
        return
    if args.server and server != args.server:
        diag = "; ".join(f"{u}: {d}" for u, d in tried)
        print(f"FEHLER: --server {args.server} ist nicht die Projekt-Instanz ({diag}) — "
              f"kein Submit an eine Fremdinstanz.", file=sys.stderr)
        sys.exit(3)
    if not server:
        diag = "\n".join(f"  {u}: {d}" for u, d in tried)
        print(f"FEHLER: Projekt-ComfyUI (v{comfy_target.EXPECT_VERSION_PREFIX}.x, {comfy_target.COMFY_ROOT}) "
              f"nicht gefunden:\n{diag}\n  Starten mit: tools\\start_comfy.ps1 (Boot ~30-60s)",
              file=sys.stderr)
        sys.exit(3)
    args.server = server
    print(f"Server: {server} (Projekt-Fingerprint bestaetigt)")

    client_id = str(uuid.uuid4())
    t_submit = time.time()
    try:
        resp = http_json(args.server + "/prompt", {"prompt": prompt, "client_id": client_id})
    except urllib.error.HTTPError as e:  # MUSS vor URLError stehen (Subklasse!)
        body = e.read().decode("utf-8", errors="replace")
        print(f"SUBMIT ABGELEHNT (HTTP {e.code}):", file=sys.stderr)
        try:
            err = json.loads(body)
            print(json.dumps(err.get("error", err), indent=1)[:2000], file=sys.stderr)
            for nid, ne in (err.get("node_errors") or {}).items():
                for detail in ne.get("errors", []):
                    print(f"  Node {nid}: {detail.get('message')} — {detail.get('details')}",
                          file=sys.stderr)
        except ValueError:
            print(body[:2000], file=sys.stderr)
        sys.exit(4)
    except (urllib.error.URLError, OSError) as e:
        # Server zwischen /queue-Check und POST weggefallen
        print(f"FEHLER: ComfyUI beim Submit nicht erreichbar unter {args.server} ({e}).",
              file=sys.stderr)
        sys.exit(3)

    pid = resp.get("prompt_id")
    if resp.get("node_errors"):
        print("SUBMIT mit Node-Fehlern abgelehnt:", file=sys.stderr)
        print(json.dumps(resp["node_errors"], indent=1)[:2000], file=sys.stderr)
        sys.exit(4)
    print(f"Submitted: prompt_id={pid} (Queue-Nr. {resp.get('number')})")
    if args.no_wait:
        print(f"Poll spaeter: GET {args.server}/history/{pid}")
        return

    t0 = time.time()
    try:
        log_pos = os.path.getsize(COMFY_LOG)  # tqdm-Zeilen des VORJOBS ausblenden
    except OSError:
        log_pos = 0
    last_line = ""
    while time.time() - t0 < args.timeout:
        time.sleep(2.0)
        try:
            hist = http_json(args.server + f"/history/{pid}")
        except (urllib.error.URLError, OSError):
            continue  # Server busy/kurz weg — weiter pollen
        entry = hist.get(pid)
        if entry:
            status = entry.get("status", {})
            ok = status.get("status_str") == "success" and status.get("completed")
            print()  # Zeile nach Progress-Ticker abschliessen
            if ok:
                files = collect_outputs(entry)
                problems = check_pose_postcondition(prompt, files, t_submit)
                problems += check_video_postconditions(prompt, files)
                if problems:
                    for p in problems:
                        print("PROBLEM: " + p, file=sys.stderr)
                    print("JOB LIEF DURCH, aber Postcondition verletzt — "
                          "Ergebnis NICHT verwenden.", file=sys.stderr)
                    sys.exit(1)
                print(f"ERFOLG nach {time.time() - t0:.0f}s. Outputs:")
                for p in files or ["(keine Dateien in der History gemeldet)"]:
                    print(f"  {p}")
                return
            print_job_error(status)
            if not any(m and m[0] in ("execution_error", "execution_interrupted")
                       for m in status.get("messages", [])):
                # Kein stummes Exit 1 (die Fehlerklasse, die dieses Tool abschaffen soll)
                print(f"JOB NICHT ERFOLGREICH ohne Fehlermeldung in der History: "
                      f"status_str={status.get('status_str')!r} completed={status.get('completed')!r}\n"
                      f"  Roh-Status: {json.dumps(status)[:500]}", file=sys.stderr)
            sys.exit(1)
        prog = tail_progress(log_pos)
        line = f"  laeuft {time.time() - t0:5.0f}s" + (f" | {prog}" if prog else "")
        if line != last_line:
            print(line)
            last_line = line

    print(f"\nTIMEOUT nach {args.timeout}s — Job laeuft evtl. noch: "
          f"GET {args.server}/history/{pid}", file=sys.stderr)
    sys.exit(5)


if __name__ == "__main__":
    main()
