"""BotterDancer HUD — lokaler Produktions-Dashboard-Server.

Ein einziger stdlib-Server (kein pip-Paket noetig), der den echten Systemzustand
aggregiert und als HUD unter http://127.0.0.1:8189 anzeigt:

- GPU: VRAM/Util/Temp/Power via nvidia-smi (NICHT /system_stats — das luegt unter WDDM,
  siehe REVIEW-SPIKE.md) + Prozessliste mit Fremdprozess-Erkennung (Ollama/llama-server
  = die live gemessene 5x-Slowdown-Ursache).
- ComfyUI: online?, Queue, Job-Fortschritt (tqdm-Zeilen aus comfyui_utf8.log geparst:
  Step x/y, s/it, ETA), letzte Jobs aus /history mit Dauer + Status.
- System-Checks: Modelle/venvs/Assets/Disk/Launch-Flags (--highvram-Falle).
- Outputs: neueste Render mit Video-Vorschau (Range-Requests fuer Seeking).
- Aktionen (User-Klick): ComfyUI starten, /free, /interrupt, Ollama entladen.

Start:  tools/start_dashboard.ps1   (oder: python tools/dashboard.py)
"""
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
import urllib.parse

import comfy_target  # gemeinsame Projekt-Instanz-Definition (Audit 04.08., F1)

# --- Konfiguration -----------------------------------------------------------
COMFY_ROOT = r"C:\ComfyUI"
COMFY_LOG = os.path.join(COMFY_ROOT, "comfyui_utf8.log")
OUTPUT_DIR = os.path.join(COMFY_ROOT, "output")
INPUT_DIR = os.path.join(COMFY_ROOT, "input")
GVHMR_ROOT = r"C:\GVHMR"
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8189

# Audit 04.08. F1: Port 8188 kann von der ComfyUI-DESKTOP-App belegt sein —
# nie blind dorthin. comfy_url() liefert die per Fingerprint bestaetigte
# Projekt-Instanz (oder None), mit 15s-Cache gegen Dauer-Polling.
_COMFY_URL_CACHE = {"url": None, "ts": 0.0}


def comfy_url():
    now = time.time()
    if now - _COMFY_URL_CACHE["ts"] > 15:
        url, _tried = comfy_target.find_project_server(preferred=_COMFY_URL_CACHE["url"])
        _COMFY_URL_CACHE.update({"url": url, "ts": now})
    return _COMFY_URL_CACHE["url"]


# Gemessene saubere Basis (GO-CHECKLIST Schritt 4, 10.07.): 15,7 s/it @
# 512x768x81/20 Steps. Die frueheren "8-10" galten fuer den kleineren
# 480p-Workload und erzeugten hier False-Positive-Fremdlast-Alarme (Audit F8).
BASELINE_S_IT = 15.7
# Draft-Lane (lightx2v, 4 Steps): gemessen 4-8 s/it (19.07.-04.08.) — eigene
# Basis, sonst wuerde reale Contention im Draft-Betrieb NIE alarmiert
# (Audit F8, Verifier-Note: Ein-Wert-Baseline ist im Draft-Betrieb blind).
DRAFT_BASELINE_S_IT = 8.0
_REF_WORKLOAD_PIX = 512 * 768 * 81  # Bezugsgroesse der gemessenen Baselines


def workload_from_prompt(graph):
    """Workload-Signatur aus einem laufenden Prompt-Graphen (queue_running[0][2])."""
    if not isinstance(graph, dict):
        return None
    w = h = length = steps = None
    draft = False
    for n in graph.values():
        if not isinstance(n, dict):
            continue
        ins = n.get("inputs", {})
        ct = n.get("class_type")
        if ct == "WanAnimateToVideo":
            for key, var in (("width", "w"), ("height", "h"), ("length", "l")):
                v = ins.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if var == "w":
                        w = v
                    elif var == "h":
                        h = v
                    else:
                        length = v
        elif ct == "KSampler":
            v = ins.get("steps")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                steps = v
        elif ct == "LoraLoaderModelOnly" and "lightx2v" in str(ins.get("lora_name", "")).lower():
            draft = True
    if not (w and h and length):
        return None
    return {"width": int(w), "height": int(h), "length": int(length),
            "steps": steps, "draft": draft}


def baseline_for(workload):
    """s/it-Baseline fuer den konkreten Workload; skaliert linear mit
    Pixel x Frames relativ zur gemessenen 512x768x81-Basis."""
    if not workload:
        return BASELINE_S_IT
    base = DRAFT_BASELINE_S_IT if workload.get("draft") else BASELINE_S_IT
    scale = (workload["width"] * workload["height"] * workload["length"]) / float(_REF_WORKLOAD_PIX)
    return max(base * scale, 1.0)
VRAM_CEILING_MIB = 87000  # ~85 GiB Admission-Ceiling aus dem Architektur-Review
FOREIGN_PROC_MIB = 1500   # ab hier gilt ein Fremdprozess als GPU-relevant

TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?([\d.]+)\s*(s/it|it/s)\]")

STATE = {"ts": 0, "gpu": None, "comfy": None, "progress": None,
         "checks": [], "outputs": [], "alerts": []}
STATE_LOCK = threading.Lock()
FIRST_SEEN = {}  # prompt_id -> Zeitpunkt, an dem das HUD den Job zuerst sah


def run(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "nicht gefunden: %s" % cmd[0]
    except subprocess.TimeoutExpired:
        return 124, "", "Timeout"


# --- Collector: GPU ----------------------------------------------------------
# Unter WDDM liefert nvidia-smi fuer JEDEN Prozess used_memory="[N/A]" (live
# verifiziert, Review 2026-07-19) — mem ist dann None, nie 0.0, sonst waere das
# HUD fuer Fremdlast blind. Fremd-Prozesse ohne Messwert werden NICHT einzeln
# gelistet (Re-Attack: 16 GUI-Apps als "fremd" auf leerer GPU = Vertrauensverlust),
# sondern als eine Sammelzeile gezaehlt; echte Fremdlast erkennt unter WDDM der
# s/it-Slowdown-Alert plus der globale VRAM-Wert (beide funktionieren dort).


def _num(v):
    """nvidia-smi-Feld -> float oder None ("[N/A]", "[Unknown Error]", leer)."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def gpu_snapshot():
    rc, out, err = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                        "--format=csv,noheader,nounits"])
    if rc != 0:
        return {"error": err or "nvidia-smi fehlgeschlagen"}
    lines = out.strip().splitlines()
    if not lines:
        return {"error": "nvidia-smi: leere Ausgabe"}
    vals = [v.strip() for v in lines[0].split(",")]
    if len(vals) < 6:
        return {"error": "nvidia-smi: unerwartetes Format: " + lines[0][:120]}
    mem_used, mem_total = _num(vals[0]), _num(vals[1])
    if mem_used is None or mem_total is None:
        return {"error": "nvidia-smi: VRAM-Werte nicht lesbar (Treiber-Reset?)"}
    gpu = {"mem_used_mib": mem_used, "mem_total_mib": mem_total,
           "util": _num(vals[2]), "temp": _num(vals[3]),
           "power_w": _num(vals[4]), "power_limit_w": _num(vals[5]),
           "procs": [], "gui_ctx": 0}
    rc, out, _ = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"])
    if rc == 0:
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            pid, name, mem = parts[0], parts[1], parts[2]
            low = name.lower()
            base = os.path.basename(low)
            if "comfyui" in low:
                kind = "comfy"
            elif "gvhmr" in low:
                kind = "gvhmr"
            elif any(k in low for k in ("ollama", "llama", "lm studio", "lmstudio")):
                kind = "llm"
            else:
                kind = "fremd"
            mem_f = _num(mem)  # None unter WDDM — fuer ALLE Prozesse, nicht nur Graphics
            if kind == "fremd":
                if mem_f is None:          # WDDM: nicht messbar -> Sammelzeile statt Liste
                    gpu["gui_ctx"] += 1
                    continue
                if mem_f < 200:            # Nicht-WDDM: 0-MiB-Rauschen
                    continue
            gpu["procs"].append({"pid": pid, "name": os.path.basename(name),
                                 "path": name, "mem_mib": mem_f, "kind": kind})
    return gpu


# --- Collector: ComfyUI ------------------------------------------------------
def http_json(url, payload=None, timeout=3):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}


def comfy_status():
    st = {"online": False, "queue_running": 0, "queue_pending": 0,
          "running_id": None, "running_elapsed_s": None, "history": [],
          "url": None, "workload": None}
    url = comfy_url()
    if not url:
        return st
    st["url"] = url
    try:
        q = http_json(url + "/queue")
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        return st
    st["online"] = True
    running = q.get("queue_running", [])
    st["queue_running"] = len(running)
    st["queue_pending"] = len(q.get("queue_pending", []))
    # FIRST_SEEN auf laufende Jobs beschraenken (sonst monotoner Speicher-Leak im Dauerlauf)
    running_ids = {e[1] for e in running if len(e) > 1}
    for k in list(FIRST_SEEN):
        if k not in running_ids:
            del FIRST_SEEN[k]
    if running:
        pid = running[0][1]
        st["running_id"] = pid
        FIRST_SEEN.setdefault(pid, time.time())
        st["running_elapsed_s"] = round(time.time() - FIRST_SEEN[pid])
        if len(running[0]) > 2:
            st["workload"] = workload_from_prompt(running[0][2])
    try:
        hist = http_json(url + "/history?max_items=8")
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        hist = {}
    items = []
    for hid, h in hist.items():
        status = h.get("status", {})
        ok = status.get("status_str") == "success" and status.get("completed", False)
        t_start = t_end = None
        err = None
        for msg in status.get("messages", []):
            if msg[0] == "execution_start":
                t_start = msg[1].get("timestamp")
            elif msg[0] in ("execution_success", "execution_error", "execution_interrupted"):
                t_end = msg[1].get("timestamp")
            if msg[0] == "execution_error":
                err = (msg[1].get("exception_message") or "")[:200]
        dur = round((t_end - t_start) / 1000.0) if (t_start and t_end) else None
        files = []
        for node_out in h.get("outputs", {}).values():
            for key in ("gifs", "images"):
                for f in node_out.get(key, []):
                    if f.get("filename"):
                        files.append(f["filename"])
        items.append({"id": hid[:8], "ok": ok, "duration_s": dur,
                      "error": err, "files": files[:3],
                      "t_end": t_end or t_start or 0})
    items.sort(key=lambda x: x["t_end"], reverse=True)
    st["history"] = items[:8]
    return st


def log_progress():
    """tqdm-Fortschritt aus dem UTF-8-Log (cmd-Redirect aus start_comfy.ps1)."""
    try:
        size = os.path.getsize(COMFY_LOG)
        with open(COMFY_LOG, "rb") as f:
            f.seek(max(0, size - 65536))
            chunk = f.read().decode("utf-8", errors="replace")
        age = time.time() - os.path.getmtime(COMFY_LOG)
    except OSError:
        return None
    # Nur tqdm-Zeilen NACH dem letzten Job-Marker werten: "Prompt executed" steht
    # nach dem Ende jedes Jobs, "got prompt" beim Queuen. Sonst wird die letzte
    # tqdm-Zeile des VORHERIGEN Jobs dem frisch gestarteten zugerechnet (falscher
    # Fortschritt + falscher Slowdown-Alert waehrend der Model-Load-Phase).
    # Zeilenverankert (Re-Attack): ein "got prompt" INNERHALB einer Log-/Promptzeile
    # darf den Fortschritt eines laufenden Renders nicht verschlucken.
    cut = -1
    for m in re.finditer(r"(?m)^(got prompt\s*$|Prompt executed in )", chunk):
        cut = m.start()
    if cut != -1:
        chunk = chunk[cut:]
    lines = re.split(r"[\r\n]+", chunk)
    last = None
    for line in lines:
        m = TQDM_RE.search(line)
        if m:
            last = m
    if not last:
        return None
    cur, total = int(last.group(1)), int(last.group(2))
    rate = float(last.group(3))
    s_it = rate if last.group(4) == "s/it" else (1.0 / rate if rate > 0 else 0.0)
    eta = round((total - cur) * s_it) if s_it > 0 else None
    return {"step": cur, "total": total, "s_it": round(s_it, 2),
            "eta_s": eta, "log_age_s": round(age)}


# --- Collector: Checks (12s-Kadenz) ------------------------------------------
def comfy_cmdline():
    rc, out, _ = run(["powershell", "-NoProfile", "-Command",
                      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                      "Where-Object { $_.CommandLine -like '*main.py*' -and $_.CommandLine -like '*ComfyUI*' } | "
                      "Select-Object -ExpandProperty CommandLine"], timeout=15)
    return out.strip() if rc == 0 else ""


def newest(pattern_dir, suffixes, limit=10):
    entries = []
    try:
        for name in os.listdir(pattern_dir):
            p = os.path.join(pattern_dir, name)
            if os.path.isfile(p) and name.lower().endswith(suffixes):
                st = os.stat(p)
                entries.append({"name": name, "size_mb": round(st.st_size / 1e6, 1),
                                "mtime": int(st.st_mtime)})
    except OSError:
        return []
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries[:limit]


def find_hmr4d_results():
    hits = []
    base = os.path.join(GVHMR_ROOT, "outputs")
    for root, dirs, files in os.walk(base):
        if root.count(os.sep) - base.count(os.sep) > 3:
            dirs[:] = []
            continue
        for f in files:
            if f == "hmr4d_results.pt":
                p = os.path.join(root, f)
                hits.append((os.path.getmtime(p), p))
    hits.sort(reverse=True)
    return hits[0][1] if hits else None


def build_checks(comfy_online):
    checks = []

    def add(cid, label, ok, detail="", level="critical"):
        checks.append({"id": cid, "label": label, "ok": bool(ok),
                       "detail": detail, "level": level if not ok else "good"})

    url = comfy_url()
    add("comfy_online", "Projekt-ComfyUI (Fingerprint v0.22/C:\\ComfyUI)", comfy_online,
        (url or "") if comfy_online else
        "keine Projekt-Instanz gefunden (Desktop-App auf 8188 zaehlt NICHT) — "
        "ueber Aktionen-Leiste starten", "warning")

    if comfy_online:
        cl = comfy_cmdline()
        if cl:
            bad_hv = "--highvram" in cl
            has_rv = "--reserve-vram" in cl
            add("comfy_flags", "Launch-Flags (kein --highvram, --reserve-vram)",
                (not bad_hv) and has_rv,
                ("--highvram aktiv → /free wird VRAM-No-Op! " if bad_hv else "") +
                ("--reserve-vram fehlt" if not has_rv else ""), "warning")

    wan = os.path.join(COMFY_ROOT, "models", "diffusion_models", "wan2.2_animate_14B_bf16.safetensors")
    add("model_wan", "Wan2.2-Animate-Modell", os.path.isfile(wan),
        wan if not os.path.isfile(wan) else "%.1f GB" % (os.path.getsize(wan) / 1e9))

    lora_dir = os.path.join(COMFY_ROOT, "models", "loras")
    loras = []
    try:
        loras = [f for f in os.listdir(lora_dir)
                 if any(k in f.lower() for k in ("lightx2v", "lightning", "distill"))]
    except OSError:
        pass
    add("lora_draft", "Draft-LoRA (lightx2v)", bool(loras),
        ", ".join(loras[:2]) if loras else "fehlt → keine schnelle Draft-Lane", "warning")

    gpy = os.path.join(GVHMR_ROOT, ".venv", "Scripts", "python.exe")
    add("gvhmr_venv", "GVHMR-venv", os.path.isfile(gpy), gpy if not os.path.isfile(gpy) else "")

    smplx = os.path.join(GVHMR_ROOT, "inputs", "checkpoints", "body_models", "smplx", "SMPLX_NEUTRAL.npz")
    add("smplx", "SMPL-X Body-Model", os.path.isfile(smplx),
        "" if os.path.isfile(smplx) else smplx)

    hmr = find_hmr4d_results()
    add("hmr4d", "Letzte 3D-Extraktion (hmr4d_results.pt)", hmr is not None,
        (os.path.relpath(hmr, GVHMR_ROOT) if hmr else "noch keine Extraktion"), "warning")

    for asset in ("botter_drive_center.mp4", "creature_ref.png"):
        p = os.path.join(INPUT_DIR, asset)
        add("asset_" + asset, "Asset: " + asset, os.path.isfile(p),
            "" if os.path.isfile(p) else "fehlt in ComfyUI\\input", "warning")

    add("ffmpeg", "ffmpeg im PATH", shutil.which("ffmpeg") is not None, "", "warning")

    try:
        du = shutil.disk_usage("C:\\")
        free_gb = du.free / 1e9
        add("disk", "Freier Platz C:", free_gb > 150,
            "%.0f GB frei" % free_gb, "critical" if free_gb < 50 else "warning")
    except OSError:
        pass
    return checks


# --- Alerts ------------------------------------------------------------------
def build_alerts(gpu, comfy, progress, checks):
    alerts = []
    busy = comfy and (comfy["queue_running"] > 0 or comfy["queue_pending"] > 0)

    llm_names = []
    if gpu and not gpu.get("error"):
        for p in gpu["procs"]:
            if p["kind"] == "llm":
                llm_names.append(p["name"])
                if p["mem_mib"] is None:
                    # WDDM: blosse Praesenz ist KEIN Beleg fuer Last — LM Studio haelt
                    # als Autostart-Dienst permanent einen leeren Grafikkontext
                    # (Re-Attack: bedingungsloser Alert = Dauer-Fehlalarm bei jedem
                    # Render). Nur informieren; echte Last meldet der s/it-Alert unten.
                    alerts.append({"level": "info",
                                   "text": "%s ist auf der GPU praesent (Belegung unter WDDM "
                                           "nicht messbar)" % p["name"],
                                   "action": "ollama_stop" if "ollama" in p["name"].lower() else None})
                else:
                    level = "critical" if busy else "warning"
                    txt = "%s belegt %.1f GB VRAM" % (p["name"], p["mem_mib"] / 1024)
                    if busy:
                        txt += " WAEHREND ein Render laeuft → erwartbarer 5x-Slowdown (WDDM-Spill)"
                    alerts.append({"level": level, "text": txt,
                                   "action": "ollama_stop" if "ollama" in p["name"].lower() else None})
            elif p["kind"] == "fremd" and p["mem_mib"] is not None and p["mem_mib"] >= FOREIGN_PROC_MIB:
                level = "critical" if busy else "warning"
                txt = "%s belegt %.1f GB VRAM" % (p["name"], p["mem_mib"] / 1024)
                if busy:
                    txt += " WAEHREND ein Render laeuft → erwartbarer 5x-Slowdown (WDDM-Spill)"
                alerts.append({"level": level, "text": txt, "action": None})
        if gpu["mem_used_mib"] > VRAM_CEILING_MIB:
            alerts.append({"level": "serious",
                           "text": "VRAM ueber %d GiB Ceiling (%.0f GiB) — WDDM-Spill-Gefahr (still, 10-50x langsamer)"
                                   % (VRAM_CEILING_MIB // 1024, gpu["mem_used_mib"] / 1024),
                           "action": "free"})

    if progress and comfy and comfy["queue_running"] > 0:
        base = baseline_for(comfy.get("workload"))
        if progress["s_it"] > 1.6 * base:
            wl = comfy.get("workload")
            wl_txt = ("%dx%dx%d%s" % (wl["width"], wl["height"], wl["length"],
                                      " draft" if wl.get("draft") else "")) if wl else "Standard"
            txt = ("Render laeuft mit %.1f s/it — %.1fx langsamer als Baseline (%.1f, %s)."
                   % (progress["s_it"], progress["s_it"] / base, base, wl_txt))
            # Gemessene Verlangsamung + LLM praesent = wahrscheinlicher Verursacher
            # (das ist der belastbare WDDM-Detektor, nicht die blosse Praesenz).
            if llm_names:
                txt += " Wahrscheinliche Ursache: %s auf der GPU." % ", ".join(llm_names[:2])
            else:
                txt += " Fremdlast auf der GPU?"
            alerts.append({"level": "serious", "text": txt,
                           "action": "ollama_stop" if any("ollama" in n.lower() for n in llm_names) else None})

    if comfy and comfy["history"]:
        last = comfy["history"][0]
        if last["error"]:
            alerts.append({"level": "critical",
                           "text": "Letzter Job fehlgeschlagen: %s" % last["error"], "action": None})

    for c in checks:
        if not c["ok"] and c["level"] == "critical":
            alerts.append({"level": "critical", "text": c["label"] + " fehlt: " + (c["detail"] or ""),
                           "action": None})
    return alerts


# --- Hintergrund-Refresher ---------------------------------------------------
def refresher():
    slow_ts = 0
    checks, outputs = [], []
    while True:
        try:
            gpu = gpu_snapshot()
            comfy = comfy_status()
            progress = log_progress()
            if time.time() - slow_ts > 12:
                checks = build_checks(comfy["online"])
                outputs = newest(OUTPUT_DIR, (".mp4", ".webm", ".png", ".json"), limit=10)
                slow_ts = time.time()
            alerts = build_alerts(gpu, comfy, progress, checks)
            with STATE_LOCK:
                STATE.update({"ts": int(time.time()), "gpu": gpu, "comfy": comfy,
                              "progress": progress, "checks": checks,
                              "outputs": outputs, "alerts": alerts,
                              "baseline_s_it": baseline_for(comfy.get("workload"))})
                STATE.pop("collector_error", None)  # transienter Fehler ist vorbei
                STATE.pop("error_ts", None)
        except Exception as e:  # HUD darf nie sterben
            with STATE_LOCK:
                STATE["collector_error"] = repr(e)
                STATE["error_ts"] = int(time.time())  # ts bleibt = letzte GUTE Daten
        time.sleep(2.5)


# --- Aktionen ----------------------------------------------------------------
def do_action(name):
    # /free und /interrupt NUR gegen die per Fingerprint bestaetigte
    # Projekt-Instanz — nie gegen die Desktop-App auf 8188 (Audit 04.08. F1).
    if name == "free":
        url = comfy_url()
        if not url:
            return False, "Projekt-Instanz nicht gefunden — Desktop-App wird NICHT angesprochen"
        try:
            http_json(url + "/free", {"unload_models": True, "free_memory": True})
            return True, "Modelle entladen + VRAM freigegeben (/free auf %s)" % url
        except Exception as e:
            return False, "Projekt-ComfyUI nicht erreichbar: %r" % e
    if name == "interrupt":
        url = comfy_url()
        if not url:
            return False, "Projekt-Instanz nicht gefunden — Desktop-App wird NICHT angesprochen"
        try:
            http_json(url + "/interrupt", {})
            return True, "Interrupt gesendet (%s)" % url
        except Exception as e:
            return False, "Projekt-ComfyUI nicht erreichbar: %r" % e
    if name == "start_comfy":
        if comfy_url():
            return False, "Projekt-ComfyUI laeuft bereits (%s) — kein Doppelstart" % comfy_url()
        if comfy_cmdline():
            return False, "Ein ComfyUI-Prozess existiert bereits (bootet vermutlich noch, 30-60s)"
        script = os.path.join(PROJECT, "tools", "start_comfy.ps1")
        if not os.path.isfile(script):
            return False, "start_comfy.ps1 nicht gefunden"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script],
                                creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)  # kurzer Health-Check: ParserError/Guard-Abbruch stirbt sofort
        if proc.poll() is not None:
            return False, ("start_comfy.ps1 sofort beendet (Exit %s) — laeuft ComfyUI schon, "
                           "oder Skriptfehler? Manuell pruefen." % proc.returncode)
        return True, "ComfyUI-Start ausgeloest (Boot dauert ~30-60s, Log: comfyui_utf8.log)"
    if name == "ollama_stop":
        rc, out, err = run(["ollama", "ps"], timeout=10)
        if rc != 0:
            return False, "ollama ps fehlgeschlagen: " + (err or str(rc))
        models = [l.split()[0] for l in out.strip().splitlines()[1:] if l.strip()]
        if not models:
            return True, "Kein Ollama-Modell geladen"
        msgs = []
        for m in models:
            rc2, _, err2 = run(["ollama", "stop", m], timeout=15)
            msgs.append("%s: %s" % (m, "entladen" if rc2 == 0 else ("Fehler " + err2)))
        return True, "; ".join(msgs)
    return False, "Unbekannte Aktion: " + name


# --- HTTP-Handler ------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
            try:
                with open(page, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, b"dashboard.html fehlt", "text/plain")
            return
        if self.path.startswith("/api/status"):
            with STATE_LOCK:
                body = json.dumps(STATE).encode("utf-8")
            self._send(200, body)
            return
        if self.path.startswith("/file/"):
            self._serve_file()
            return
        self._send(404, b"{}")

    def _serve_file(self):
        name = os.path.basename(urllib.parse.unquote(self.path[len("/file/"):]))
        p = os.path.join(OUTPUT_DIR, name)
        if not (os.path.isfile(p) and name.lower().endswith((".mp4", ".webm", ".png", ".jpg", ".json"))):
            self._send(404, b"not found", "text/plain")
            return
        size = os.path.getsize(p)
        ctype = {"mp4": "video/mp4", "webm": "video/webm", "png": "image/png",
                 "jpg": "image/jpeg", "json": "application/json"}[name.rsplit(".", 1)[1].lower()]
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        extra = {"Accept-Ranges": "bytes"}
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                    code = 206
                elif m.group(2) and int(m.group(2)) > 0:
                    # Suffix-Range "bytes=-N": letzte N Bytes (RFC 9110) — Browser
                    # brauchen das fuer das moov-Atom am MP4-Ende (Seeking).
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                    code = 206
                # sonst ("bytes=-"/"bytes=-0"): invalide Range ignorieren -> 200, ganze Datei
                if code == 206:
                    if start >= size or start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */%d" % size)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    extra["Content-Range"] = "bytes %d-%d/%d" % (start, end, size)
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, BrokenPipeError):
                    return
                remaining -= len(chunk)

    def do_POST(self):
        # CSRF-Schutz: beliebige Webseiten koennen sonst per no-cors-fetch auf
        # 127.0.0.1:8189 z.B. einen laufenden Render abbrechen (/interrupt).
        allowed = {"127.0.0.1:%d" % PORT, "localhost:%d" % PORT}
        origin = self.headers.get("Origin")
        if (self.headers.get("Host", "") not in allowed
                or (origin is not None and origin not in {"http://" + h for h in allowed})):
            self._send(403, b'{"ok": false, "detail": "Aktion verweigert (CSRF-Schutz: fremder Origin/Host)"}')
            return
        if self.path.startswith("/api/action/"):
            name = self.path.rsplit("/", 1)[1]
            ok, detail = do_action(name)
            self._send(200, json.dumps({"ok": ok, "detail": detail}).encode("utf-8"))
            return
        self._send(404, b"{}")


def main():
    threading.Thread(target=refresher, daemon=True).start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("BotterDancer HUD laeuft: http://127.0.0.1:%d" % PORT)
    srv.serve_forever()


if __name__ == "__main__":
    main()
