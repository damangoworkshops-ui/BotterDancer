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
        stem = os.path.splitext(os.path.basename(raw))[0][:40]
        cand = os.path.join(PROJECT, "assets", "audio", f"{stem}_segment.wav")
        if os.path.isfile(cand):
            grid = cand + ".beats.json"
    bpm = a.get("bpm") or bpm or 100.0
    synth = os.path.join(PROJECT, "assets", "audio", f"{job.name}_synth.wav")
    job.run([os.path.join(TOOLS, "make_beat_track.py"), "--bpm", f"{bpm:.2f}",
             "--duration", str(s.get("duration", 6.0)), "--out", synth],
            f"Rechtefreier Track ({bpm:.1f} BPM)")
    return {"orig_segment": (grid[:-len(".beats.json")] if grid else None),
            "grid": grid, "synth": synth, "bpm": bpm}


def step_pose_static(job, src):
    """2D-Pfad: DWPose auf das Segment, dann Multi-Person-Tracking."""
    n = len(job.s["cast"])
    job.run([os.path.join(TOOLS, "submit_workflow.py"),
             os.path.join(PROJECT, "workflows", "save_pose_81.json"),
             "--set", f"1.video={os.path.basename(src)}",
             "--set", f"3.filename_prefix={job.name}_pose", "--timeout", "900"],
            "Pose-Extraktion (DWPose)")
    pose_json = os.path.join(COMFY_OUT, f"{job.name}_pose_00001.json")
    outdir = os.path.join(COMFY_IN, f"{job.name}_crew")
    args = [os.path.join(TOOLS, "crew_pose.py"), "--src", pose_json,
            "--outdir", outdir, "--crew", str(n)]
    if n > 1:
        args.append("--split")
    job.run(args, f"Tracking ({n} Figur{'en' if n > 1 else ''})")
    return [f"{outdir}_t{i}" for i in range(n)] if n > 1 else [outdir]


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
    wf = ("wan_animate_draft.json" if s.get("quality", "final") == "draft"
          else "wan_animate_final_v2.json")
    outs = []
    for i, (pd, c) in enumerate(zip(pose_dirs, s["cast"])):
        prefix = f"{job.name}_fig{i}"
        job.run([os.path.join(TOOLS, "submit_workflow.py"),
                 os.path.join(PROJECT, "workflows", wf),
                 "--set", f"20.directory={pd}",
                 "--set", f"33.filename_prefix={prefix}",
                 "--set", f"17.image={c['ref']}",
                 "--set", f"13.text={c['prompt']}",
                 "--set", f"14.text={c.get('negative', s.get('negative', 'low quality, blurry, distorted, deformed hands, extra limbs, watermark, text'))}",
                 "--set", f"30.width={s.get('width', 1024)}",
                 "--set", f"30.height={s.get('height', 576)}", "--timeout", "1800"],
                f"Render Figur {i + 1}/{len(pose_dirs)} ({c['ref']})")
        outs.append(os.path.join(COMFY_OUT, f"{prefix}_00001.mp4"))
    return outs


def step_composite(job, clips, pose_dirs):
    if len(clips) == 1:
        return clips[0]
    out = os.path.join(COMFY_OUT, f"{job.name}_crew.mp4")
    args = [os.path.join(TOOLS, "crew_composite.py"), "--base", clips[0],
            "--overlay"] + clips[1:] + ["--overlay-pose"] + pose_dirs[1:] + \
           ["--thresh", str(job.s.get("composite_thresh", 22)), "--out", out]
    job.run(args, f"Compositing ({len(clips)} Figuren)")
    return out


def step_interpolate(job, clip):
    prefix = f"{job.name}_32"
    job.run([os.path.join(TOOLS, "submit_workflow.py"),
             os.path.join(PROJECT, "workflows", "rife_32fps.json"),
             "--set", f"1.video={clip}", "--set", f"3.filename_prefix={prefix}",
             "--timeout", "900"], "Interpolation auf 32 fps")
    return os.path.join(COMFY_OUT, f"{prefix}_00001.mp4")


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
    merged = step_composite(job, clips, pose_dirs)
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
