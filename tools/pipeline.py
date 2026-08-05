#!/usr/bin/env python
"""BotterDancer — Pipeline-Runner: ein Job-Rezept statt 19 Handgriffe.

Loest das P2-Item aus AUDIT-2026-07-19 und macht die beiden Kamera-Modi
explizit, die bisher nur als unterschiedliche Kommandoketten existierten:

  camera "static"  — 2D-Pfad (DWPose). Die Pose-Sequenz TRAEGT die
                     Original-Kamerafuehrung bereits, sie wird also 1:1
                     uebernommen. Kann MEHRERE Taenzerinnen (Full-Crew).
  camera "moving"  — 3D-Pfad (GVHMR + Reprojektion). Frei waehlbare virtuelle
                     Kamerafahrt (Steadicam/Gimbal/Crane) inkl. Beat-Warp und
                     Foot-Anchoring. Nur EINE Person (GVHMR-Limit).

Die Kopplung ist echt und keine Bequemlichkeit: Wer die Kamera frei fuehren
will, gibt Gruppen auf — und umgekehrt.

  python pipeline.py --job jobs/mein_job.json [--dry-run] [--stop-after pose]

Rezept (JSON), Minimalbeispiel static:
  {"name": "crew_demo", "source_video": "E:\\\\clip.mp4", "start": 441.0,
   "duration": 6.2, "camera": "static", "quality": "final",
   "cast": [{"ref": "ref_neon.png", "prompt": "a dancer ..."}]}

Jeder Schritt meldet sich laut, bricht bei Fehler ab und schreibt am Ende ein
Job-Protokoll neben den Output (Reproduzierbarkeit, Op-Log aus dem Review).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(TOOLS)
PY_COMFY = r"C:\ComfyUI\venv\Scripts\python.exe"
PY_GVHMR = r"C:\GVHMR\.venv\Scripts\python.exe"
PY_AUDIO = os.path.join(PROJECT, ".venv-audio", "Scripts", "python.exe")
COMFY_IN = r"C:\ComfyUI\input"
COMFY_OUT = r"C:\ComfyUI\output"
FFMPEG_FALLBACK_DIR = (r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages"
                       r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                       r"\ffmpeg-8.1-full_build\bin")
STEPS = ("segment", "audio", "pose", "render", "composite", "interpolate", "export")


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(FFMPEG_FALLBACK_DIR, name + ".exe")
    return cand if os.path.isfile(cand) else None


class Job:
    def __init__(self, spec, dry_run=False, stop_after=None):
        self.s = spec
        self.dry = dry_run
        self.stop_after = stop_after
        self.name = spec["name"]
        self.log = []
        self.t0 = time.time()

    def say(self, msg):
        print(f"[pipeline] {msg}", flush=True)

    def run(self, cmd, what, python=None):
        exe = python or PY_COMFY
        full = [exe] + cmd if cmd[0].endswith(".py") else cmd
        self.say(f"{what}: {' '.join(str(c) for c in full[:4])} ...")
        if self.dry:
            self.log.append({"step": what, "cmd": [str(c) for c in full], "dry": True})
            return ""
        t = time.time()
        r = subprocess.run([str(c) for c in full], capture_output=True, text=True)
        self.log.append({"step": what, "cmd": [str(c) for c in full],
                         "rc": r.returncode, "seconds": round(time.time() - t, 1)})
        if r.returncode != 0:
            print(r.stdout[-1500:], file=sys.stderr)
            print(r.stderr[-1500:], file=sys.stderr)
            self.fail(f"{what} fehlgeschlagen (Exit {r.returncode})")
        tail = "\n".join(r.stdout.strip().splitlines()[-2:])
        if tail:
            print("    " + tail.replace("\n", "\n    "))
        return r.stdout

    def fail(self, msg, code=3):
        self.say(f"ABBRUCH: {msg}")
        self.write_log(failed=msg)
        sys.exit(code)

    def should_stop(self, step):
        return self.stop_after == step

    def write_log(self, failed=None, outputs=None):
        p = os.path.join(COMFY_OUT, f"{self.name}.job.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"spec": self.s, "steps": self.log, "failed": failed,
                       "outputs": outputs or [],
                       "total_seconds": round(time.time() - self.t0, 1)}, f, indent=1)
        self.say(f"Protokoll: {p}")


def validate(spec):
    """Rezept pruefen BEVOR GPU-Zeit verbrannt wird."""
    errs = []
    for key in ("name", "source_video", "camera"):
        if not spec.get(key):
            errs.append(f"Pflichtfeld fehlt: {key}")
    if any(c.isspace() for c in (spec.get("name") or "")):
        # Der Name wird zum Dateinamen-Stem; song_beats ersetzt Leerzeichen
        # durch Unterstriche, die Pipeline suchte dann am falschen Ort und
        # verlor Beat-Grid und Original-Vorschau still (Review 06.08.).
        errs.append("name darf keine Leerzeichen enthalten")
    cam = spec.get("camera")
    if cam not in ("static", "moving"):
        errs.append(f"camera muss 'static' oder 'moving' sein, ist {cam!r}")
    cast = spec.get("cast") or []
    if not cast:
        errs.append("cast (mindestens eine Figur mit ref+prompt) fehlt")
    if cam == "moving" and len(cast) > 1:
        errs.append(f"camera=moving kann nur EINE Figur (GVHMR trackt eine Person), "
                    f"cast hat {len(cast)} — fuer Gruppen camera=static nutzen")
    for i, c in enumerate(cast):
        if not c.get("ref"):
            errs.append(f"cast[{i}].ref fehlt")
        elif not os.path.isfile(os.path.join(COMFY_IN, c["ref"])):
            errs.append(f"cast[{i}].ref nicht in {COMFY_IN}: {c['ref']}")
        if not c.get("prompt"):
            errs.append(f"cast[{i}].prompt fehlt")
    if spec.get("source_video") and not os.path.isfile(spec["source_video"]):
        errs.append(f"source_video nicht gefunden: {spec['source_video']}")
    if spec.get("quality", "final") not in ("draft", "final"):
        errs.append("quality muss 'draft' oder 'final' sein")
    bg = spec.get("background", "studio")
    if bg not in ("studio", "room"):
        errs.append(f"background muss 'studio' oder 'room' sein, ist {bg!r}")
    if spec.get("figure_crop") and cam == "moving":
        # Der 3D-Pfad rendert bereits figurzentriert (die virtuelle Kamera
        # bestimmt den Ausschnitt) — ein zweiter Crop waere sinnlos.
        errs.append("figure_crop ist nur mit camera=static sinnvoll (der 3D-Pfad "
                    "rahmt die Figur bereits ueber die virtuelle Kamera)")
    gate = spec.get("identity_gate", "warn")
    if gate not in ("off", "warn", "strict"):
        errs.append(f"identity_gate muss 'off', 'warn' oder 'strict' sein, ist {gate!r}")
    sim = spec.get("identity_min_sim", 0.75)
    if not isinstance(sim, (int, float)) or not 0.0 < sim < 1.0:
        errs.append(f"identity_min_sim muss zwischen 0 und 1 liegen, ist {sim!r}")
    if bg == "room" and cam == "moving":
        # Das Plate ist ein Standbild aus einer statischen Kamera; eine
        # virtuelle Fahrt haette keine passende Parallaxe.
        errs.append("background=room setzt camera=static voraus (das Raum-Plate "
                    "stammt aus einer statischen Kamera)")
    return errs


def step_segment(job):
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        job.fail("ffmpeg nicht gefunden", 2)
    s = job.s
    w, h = s.get("width", 1024), s.get("height", 576)
    dst = os.path.join(COMFY_IN, f"{job.name}_src.mp4")
    job.run([ffmpeg, "-v", "error", "-ss", str(s.get("start", 0.0)),
             "-t", str(s.get("duration", 6.2)), "-i", s["source_video"], "-an",
             "-vf", f"scale={w}:{h}", "-c:v", "libx264", "-crf", "12",
             "-pix_fmt", "yuv420p", "-y", dst], "Segment schneiden")
    return dst


def step_audio(job):
    """Beat-Grid + zwei Tonspuren: Original (nur lokale Vorschau) und
    rechtefreier Synth-Track (Export). Produktregel 04.07."""
    s = job.s
    a = s.get("audio", {})
    ffmpeg = find_tool("ffmpeg")
    raw = os.path.join(COMFY_OUT, f"{job.name}_origaudio.wav")
    job.run([ffmpeg, "-v", "error", "-ss", str(s.get("start", 0.0)),
             "-t", str(s.get("duration", 6.2) + 4), "-i", s["source_video"],
             "-vn", "-ar", "44100", "-ac", "2", "-y", raw], "Originalton extrahieren")
    grid = bpm = None
    if os.path.isfile(PY_AUDIO) and not job.dry:
        out = job.run([os.path.join(TOOLS, "song_beats.py"), "--audio", raw,
                       "--out-dir", os.path.join(PROJECT, "assets", "audio"),
                       "--start", "0.0", "--duration", str(s.get("duration", 6.0))],
                      "Beat-Analyse (Beat This!)", python=PY_AUDIO)
        for line in out.splitlines():
            if "BPM" in line and "Segment" in line:
                for tok in line.replace(",", " ").split():
                    try:
                        v = float(tok)
                        if 40 < v < 220:
                            bpm = v
                    except ValueError:
                        pass
        # song_beats bildet den Stem MIT Leerzeichen-Ersetzung — hier identisch
        # rechnen, sonst laeuft die Suche ins Leere und das Grid geht still verloren.
        stem = os.path.splitext(os.path.basename(raw))[0][:40].replace(" ", "_")
        cand = os.path.join(PROJECT, "assets", "audio", f"{stem}_segment.wav")
        if os.path.isfile(cand):
            grid = cand + ".beats.json"
        else:
            job.say(f"WARNUNG: Beat-Segment nicht gefunden ({cand}) — "
                    f"Beat-Warp und Original-Vorschau entfallen fuer diesen Lauf.")
    bpm = a.get("bpm") or bpm or 100.0
    synth = os.path.join(PROJECT, "assets", "audio", f"{job.name}_synth.wav")
    job.run([os.path.join(TOOLS, "make_beat_track.py"), "--bpm", f"{bpm:.2f}",
             "--duration", str(s.get("duration", 6.0)), "--out", synth],
            f"Rechtefreier Track ({bpm:.1f} BPM)")
    return {"orig_segment": (grid[:-len(".beats.json")] if grid else None),
            "grid": grid, "synth": synth, "bpm": bpm}


def comfy_output(job, prefix, ext, t_after):
    """Juengste ComfyUI-Ausgabe zu `prefix` seit `t_after`.

    Die Save-Nodes zaehlen hoch (_00001, _00002, ...). Ein fest verdrahtetes
    _00001 liest beim zweiten Lauf desselben Job-Namens still die ALTEN
    Dateien — Tracking, Identitaets-Check und Export arbeiten dann auf dem
    Ergebnis des ERSTEN Laufs (Stale-Falle, Review 06.08.)."""
    import glob
    fallback = os.path.join(COMFY_OUT, f"{prefix}_00001{ext}")
    if job.dry:
        return fallback
    cands = [p for p in glob.glob(os.path.join(COMFY_OUT, f"{prefix}_*{ext}"))
             if os.path.getmtime(p) >= t_after - 2]
    if not cands:
        job.fail(f"keine frische Ausgabe {prefix}_*{ext} seit dem Submit gefunden "
                 f"— ComfyUI-Ausgabeverzeichnis pruefen")
    return max(cands, key=os.path.getmtime)


def step_pose_static(job, src):
    """2D-Pfad: DWPose auf das Segment, dann Multi-Person-Tracking."""
    n = len(job.s["cast"])
    t0 = time.time()
    job.run([os.path.join(TOOLS, "submit_workflow.py"),
             os.path.join(PROJECT, "workflows", "save_pose_81.json"),
             "--set", f"1.video={os.path.basename(src)}",
             "--set", f"3.filename_prefix={job.name}_pose", "--timeout", "900"],
            "Pose-Extraktion (DWPose)")
    pose_json = comfy_output(job, f"{job.name}_pose", ".json", t0)
    job.s["_pose_json"] = pose_json
    outdir = os.path.join(COMFY_IN, f"{job.name}_crew")
    args = [os.path.join(TOOLS, "crew_pose.py"), "--src", pose_json,
            "--outdir", outdir, "--crew", str(n)]
    if n > 1:
        args.append("--split")
    job.run(args, f"Tracking ({n} Figur{'en' if n > 1 else ''})")
    dirs = [f"{outdir}_t{i}" for i in range(n)] if n > 1 else [outdir]
    job.s["_full_pose_dirs"] = dirs

    if job.s.get("figure_crop"):
        # Jede Figur bekommt ihr eigenes Fenster in voller Render-Aufloesung.
        # Der Gewinn haengt davon ab, wie klein die Figur im Bild steht — bei
        # Weitwinkel-Gruppen ist er am groessten.
        cw, chh = job.s.get("crop_width", 512), job.s.get("crop_height", 768)
        crops = []
        for i, d in enumerate(dirs):
            cd = os.path.join(COMFY_IN, f"{job.name}_crop_t{i}")
            job.run([os.path.join(TOOLS, "figure_crop.py"), "--pose-dir", d,
                     "--outdir", cd, "--width", str(cw), "--height", str(chh),
                     "--margin", str(job.s.get("crop_margin", 0.22))],
                    f"Figuren-Crop {i + 1}/{len(dirs)}")
            crops.append(cd)
        job.s["_crop_meta"] = [c + ".crop.json" for c in crops]
        return crops
    return dirs


def step_pose_moving(job, src):
    """3D-Pfad: GVHMR, optional Beat-Warp + Foot-Anchoring, dann Kamerafahrt."""
    s = job.s
    gv_out = os.path.join(r"C:\GVHMR\outputs\demo", f"{job.name}_src", "hmr4d_results.pt")
    if not os.path.isfile(gv_out) or s.get("force_gvhmr"):
        job.run(["demo.py", "--video", src, "-s"], "3D-Motion (GVHMR)", python=PY_GVHMR)
    cur = gv_out
    audio = job.s.get("_audio", {})
    if s.get("beat_warp", True) and audio.get("grid"):
        warped = cur.replace(".pt", "_warp.pt")
        job.run([os.path.join(TOOLS, "motion_warp.py"), "--src", cur,
                 "--grid", audio["grid"], "--out", warped],
                "Beat-Warp", python=PY_GVHMR)
        cur = warped
    if s.get("foot_anchor", True):
        anchored = cur.replace(".pt", "_anchor.pt")
        job.run([os.path.join(TOOLS, "foot_anchor.py"), "--src", cur, "--out", anchored],
                "Foot-Anchoring", python=PY_GVHMR)
        cur = anchored
    mv = s.get("camera_move", {})
    outdir = os.path.join(COMFY_IN, f"{job.name}_cam")
    args = [os.path.join(TOOLS, "reproject_camera.py"), "--src", cur, "--outdir", outdir,
            "--width", str(s.get("width", 1024)), "--height", str(s.get("height", 576))]
    for key, flag in (("azimuth", "--azimuth"), ("elevation", "--elevation"),
                      ("distance_scale", "--distance-scale")):
        v = mv.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            args += [flag, str(v[0]), flag + "-end", str(v[1])]
        else:
            args += [flag, str(v)]
    job.run(args, "Virtuelle Kamerafahrt", python=PY_GVHMR)
    return [outdir]


def step_render(job, pose_dirs):
    s = job.s
    draft = s.get("quality", "final") == "draft"
    wf = "wan_animate_draft.json" if draft else "wan_animate_final_v2.json"
    if draft:
        # Empirisch 06.08. am selben Track belegt: die Draft-Lane faehrt cfg 1.0,
        # dort wird der Negativ-Prompt gar nicht ausgewertet (keine CFG) — das
        # Modell erfindet dann gern eine zweite Figur zum selben Skelett. Der
        # Final-Render (cfg 4.0) mit identischem Input bleibt bei einer Figur.
        job.say("HINWEIS: quality=draft laeuft mit cfg 1.0 — Negativ-Prompts sind "
                "WIRKUNGSLOS. Draft taugt fuer Timing/Komposition, nicht zur "
                "Beurteilung von Figurenanzahl oder Artefakten.")
    # Beim Crop rendert jede Figur in ihrem eigenen Fenster, sonst in Szenengroesse
    if s.get("figure_crop"):
        rw, rh = s.get("crop_width", 512), s.get("crop_height", 768)
    else:
        rw, rh = s.get("width", 1024), s.get("height", 576)
    outs = []
    for i, (pd, c) in enumerate(zip(pose_dirs, s["cast"])):
        prefix = f"{job.name}_fig{i}"
        t0 = time.time()
        job.run([os.path.join(TOOLS, "submit_workflow.py"),
                 os.path.join(PROJECT, "workflows", wf),
                 "--set", f"20.directory={pd}",
                 "--set", f"33.filename_prefix={prefix}",
                 "--set", f"17.image={c['ref']}",
                 "--set", f"13.text={c['prompt']}",
                 "--set", f"14.text={c.get('negative', s.get('negative', 'low quality, blurry, distorted, deformed hands, extra limbs, watermark, text'))}",
                 "--set", f"30.width={rw}",
                 "--set", f"30.height={rh}", "--timeout", "1800"],
                f"Render Figur {i + 1}/{len(pose_dirs)} ({c['ref']})")
        clip = comfy_output(job, prefix, ".mp4", t0)
        outs.append(clip)
        check_identity(job, clip, c, pd, i)
    return outs


def _plate_from_first(job, clip):
    """Ohne Raum-Plate braucht der Crop-Modus trotzdem einen Hintergrund in
    Szenengroesse. Der erste Solo-Render taugt dafuer nicht (er IST ein
    Ausschnitt), deshalb ein leeres Studio-Plate in Szenenmassen."""
    ffmpeg = find_tool("ffmpeg")
    out = os.path.join(COMFY_IN, f"{job.name}_bgplate.mp4")
    w, h = job.s.get("width", 1024), job.s.get("height", 576)
    job.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
             f"color=c=0xE8EAEC:s={w}x{h}:r=16", "-frames:v", "81",
             "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-y", out],
            "Neutrales Studio-Plate (Crop-Modus ohne echten Raum)")
    return out


def _composite_pose_dirs(job, pose_dirs):
    """Fuer die Masken/Boxen im Composite immer die VOLLBILD-Posen nehmen —
    beim Crop sind `pose_dirs` die Ausschnitte, die dort nicht passen."""
    return job.s.get("_full_pose_dirs") or pose_dirs


def check_identity(job, clip, cast_entry, pose_dir, idx):
    """Traegt die gerenderte Figur wirklich ihre Referenz? (CLIP-Vision)

    Laeuft direkt nach dem Render, damit eine misslungene Figur auffaellt,
    BEVOR Compositing, Interpolation und Export darauf aufbauen. Default ist
    'warn': ein Abbruch nach neun Minuten Renderzeit hilft niemandem, eine
    uebersehene Fehlbesetzung aber auch nicht.
    """
    mode = job.s.get("identity_gate", "warn")
    if mode == "off" or job.dry:
        return
    ref = os.path.join(COMFY_IN, cast_entry["ref"])
    args = [os.path.join(TOOLS, "identity_check.py"), "--video", clip,
            "--ref", ref, "--samples", "5",
            "--min-sim", str(job.s.get("identity_min_sim", 0.75))]
    # Die Pose-Box schneidet die Figur aus dem Szenenbild — bei einem
    # figurzentrierten Crop waere das ein ZWEITER Ausschnitt und verglichen
    # wuerde ein Figur-Detail mit dem Referenz-Ganzbild (gemessen 06.08.:
    # 0.72-0.75 statt der korrekten 0.81-0.85).
    if not job.s.get("figure_crop"):
        args += ["--pose-dir", pose_dir]
    r = subprocess.run([PY_COMFY] + args, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("[ident] Mittel")), "")
    job.log.append({"step": f"Identitaets-Check Figur {idx + 1}", "rc": r.returncode,
                    "summary": line.strip()})
    if r.returncode == 0:
        job.say(f"Identitaet Figur {idx + 1} ok — {line.replace('[ident] ', '')}")
        return
    msg = (f"Figur {idx + 1} ({cast_entry['ref']}) trifft ihre Referenz nicht: "
           f"{line.replace('[ident] ', '') or r.stdout.strip()[-160:]}")
    if mode == "strict":
        job.fail(msg + " (identity_gate=strict)", 5)
    job.say("WARNUNG: " + msg + " — Rezept pruefen (Prompt/Referenz), "
            "identity_gate='strict' bricht hier ab.")


def step_room_plate(job, src, pose_json, fps=16.0):
    """Raum-Plate fuer background='room': echter Raum, Personen entfernt.
    Braucht fps-gleiches Video UND Masken aller Personen (Zeitversatz-Falle)."""
    ffmpeg = find_tool("ffmpeg")
    matched = os.path.join(COMFY_IN, f"{job.name}_src{int(fps)}.mp4")
    job.run([ffmpeg, "-v", "error", "-i", src, "-vf", f"fps={fps:g}",
             "-frames:v", "81", "-c:v", "libx264", "-crf", "10",
             "-pix_fmt", "yuv420p", "-y", matched], "Quelle auf Pose-fps bringen")
    masks = os.path.join(COMFY_IN, f"{job.name}_maskall")
    job.run([os.path.join(TOOLS, "pose_silhouette.py"), "--src", pose_json,
             "--outdir", masks, "--dilate", "34"], "Personen-Masken (alle)")
    plate = os.path.join(COMFY_IN, f"{job.name}_plate.mp4")
    job.run([os.path.join(TOOLS, "room_plate.py"), "--video", matched,
             "--masks", masks, "--fps", str(fps), "--out", plate], "Raum-Plate")
    return plate


def step_composite(job, clips, pose_dirs, plate=None):
    crop_meta = job.s.get("_crop_meta")
    if len(clips) == 1 and not plate and not crop_meta:
        return clips[0]
    out = os.path.join(COMFY_OUT, f"{job.name}_crew.mp4")
    full_dirs = _composite_pose_dirs(job, pose_dirs)
    # Mit Raum-Plate ist die Basis der echte Raum und ALLE Figuren sind Overlays;
    # ohne Plate liefert der erste Clip Hintergrund UND erste Figur. Crop-Renders
    # sind IMMER Overlays — ein Ausschnitt taugt nicht als Hintergrund.
    if plate or crop_meta:
        base = plate or _plate_from_first(job, clips[0])
        ov, ovp = clips, full_dirs
        meta = crop_meta
    else:
        base, ov, ovp = clips[0], clips[1:], full_dirs[1:]
        meta = None
    args = [os.path.join(TOOLS, "crew_composite.py"), "--base", base,
            "--overlay"] + ov + ["--overlay-pose"] + ovp + \
           ["--thresh", str(job.s.get("composite_thresh", 22)), "--out", out]
    if meta:
        args += ["--overlay-crop"] + meta
    if plate:
        # Plate und Solo-Renders haben verschiedene Studio-Helligkeiten; der
        # Angleich wuerde die Figuren ans Plate anpassen statt umgekehrt.
        args.append("--no-match")
    job.run(args, f"Compositing ({len(clips)} Figuren"
                  + (" auf echtem Raum)" if plate else ")"))
    return out


def step_interpolate(job, clip):
    prefix = f"{job.name}_32"
    t0 = time.time()
    job.run([os.path.join(TOOLS, "submit_workflow.py"),
             os.path.join(PROJECT, "workflows", "rife_32fps.json"),
             "--set", f"1.video={clip}", "--set", f"3.filename_prefix={prefix}",
             "--timeout", "900"], "Interpolation auf 32 fps")
    return comfy_output(job, prefix, ".mp4", t0)


def step_export(job, clip, audio):
    ffmpeg = find_tool("ffmpeg")
    outs = []
    if audio.get("orig_segment") and os.path.isfile(audio["orig_segment"]):
        prev_dir = os.path.join(COMFY_OUT, "preview")
        os.makedirs(prev_dir, exist_ok=True)
        prev = os.path.join(prev_dir, f"{job.name}_preview_NUR_LOKAL.mp4")
        job.run([ffmpeg, "-v", "error", "-i", clip, "-i", audio["orig_segment"],
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                 "-b:a", "192k", "-shortest", "-y", prev],
                "Lokale Vorschau mit Originalton")
        outs.append(prev)
    args = [os.path.join(TOOLS, "export_clip.py"), clip, "--audio", audio["synth"],
            "--crf", str(job.s.get("export_crf", 12)), "--overwrite"]
    if job.s.get("no_watermark"):
        args.append("--no-watermark")
    job.run(args, "Signierter Export (C2PA)")
    outs.append(os.path.join(COMFY_OUT, "export",
                             os.path.splitext(os.path.basename(clip))[0] + "_export.mp4"))
    return outs


def main():
    ap = argparse.ArgumentParser(description="BotterDancer Pipeline-Runner")
    ap.add_argument("--job", required=True, help="Job-Rezept (JSON)")
    ap.add_argument("--dry-run", action="store_true", help="nur Plan zeigen, nichts ausfuehren")
    ap.add_argument("--stop-after", choices=STEPS, help="nach diesem Schritt anhalten")
    args = ap.parse_args()

    with open(args.job, "r", encoding="utf-8-sig") as f:
        spec = json.load(f)
    errs = validate(spec)
    if errs:
        for e in errs:
            print("FEHLER: " + e, file=sys.stderr)
        return 2

    job = Job(spec, args.dry_run, args.stop_after)
    cam = spec["camera"]
    job.say(f"Job '{job.name}': camera={cam}, {len(spec['cast'])} Figur(en), "
            f"quality={spec.get('quality', 'final')}"
            + (" [DRY-RUN]" if args.dry_run else ""))

    src = step_segment(job)
    if job.should_stop("segment"):
        job.write_log(outputs=[src])
        return 0
    audio = step_audio(job)
    spec["_audio"] = audio
    if job.should_stop("audio"):
        job.write_log(outputs=[audio.get("synth")])
        return 0

    pose_dirs = step_pose_static(job, src) if cam == "static" else step_pose_moving(job, src)
    if job.should_stop("pose"):
        job.write_log(outputs=pose_dirs)
        return 0

    clips = step_render(job, pose_dirs)
    if job.should_stop("render"):
        job.write_log(outputs=clips)
        return 0
    plate = None
    if spec.get("background") == "room":
        plate = step_room_plate(job, src, spec["_pose_json"])
    merged = step_composite(job, clips, pose_dirs, plate)
    if job.should_stop("composite"):
        job.write_log(outputs=[merged])
        return 0
    smooth = step_interpolate(job, merged)
    if job.should_stop("interpolate"):
        job.write_log(outputs=[smooth])
        return 0
    finals = step_export(job, smooth, audio)

    job.say(f"FERTIG nach {(time.time() - job.t0) / 60:.1f} min:")
    for f in finals:
        job.say(f"  {f}")
    job.write_log(outputs=finals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
