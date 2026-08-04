#!/usr/bin/env python
"""BotterDancer — Multi-Angle-Beat-Cut.

Schneidet zwischen mehreren Renders DERSELBEN Choreo-Timeline (die fuenf
Kamerawinkel laufen synchron) auf dem Beat-Grid eines Tracks — der klassische
Tanzvideo-Schnitt, hier deterministisch. Vorstufe des Choreographers:
Beat-Grid rein (aus make_beat_track.py-Sidecar oder --bpm), Schnittliste raus.

  python beat_cut.py --videos a.mp4 b.mp4 c.mp4 --grid beat.wav.beats.json
                     --every 2 --out cut.mp4

Erwartet CFR-Inputs gleicher fps/Aufloesung (ffprobe-geprueft, fail-closed);
Schnittpunkte werden aufs Frame-Raster gerundet. Output ist stumm — Audio
kommt erst im terminalen Export dazu (export_clip.py --audio).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction

FFMPEG_FALLBACK_DIR = (r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages"
                       r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                       r"\ffmpeg-8.1-full_build\bin")


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(FFMPEG_FALLBACK_DIR, name + ".exe")
    return cand if os.path.isfile(cand) else None


def fail(msg, code=2):
    print("FEHLER: " + msg, file=sys.stderr)
    sys.exit(code)


def probe(ffprobe, path):
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=avg_frame_rate,nb_frames,width,height",
                        "-of", "json", path], capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"ffprobe scheitert an {path}", 3)
    s = json.loads(r.stdout)["streams"][0]
    return {"fps": Fraction(s["avg_frame_rate"]), "frames": int(s["nb_frames"]),
            "w": int(s["width"]), "h": int(s["height"])}


def main():
    ap = argparse.ArgumentParser(description="Multi-Angle-Beat-Cut (synchrone Timelines)")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="Winkel-Renders in Schnitt-Reihenfolge (wird zyklisch durchlaufen)")
    ap.add_argument("--grid", help="Beat-Grid-JSON (make_beat_track.py-Sidecar)")
    ap.add_argument("--bpm", type=float, help="alternativ zu --grid: festes Tempo, Offset 0")
    ap.add_argument("--every", type=int, default=2, help="Winkelwechsel alle N Beats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=14)
    args = ap.parse_args()

    if bool(args.grid) == bool(args.bpm):
        fail("genau EINE Quelle angeben: --grid ODER --bpm")
    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        fail("ffmpeg/ffprobe nicht gefunden")
    for v in args.videos:
        if not os.path.isfile(v):
            fail(f"Video fehlt: {v}")

    infos = [probe(ffprobe, v) for v in args.videos]
    ref = infos[0]
    for v, i in zip(args.videos, infos):
        if (i["fps"], i["w"], i["h"]) != (ref["fps"], ref["w"], ref["h"]):
            fail(f"{os.path.basename(v)}: {i['w']}x{i['h']}@{i['fps']} passt nicht zu "
                 f"Referenz {ref['w']}x{ref['h']}@{ref['fps']} — Inputs muessen homogen sein.")
    fps = float(ref["fps"])
    n_frames = min(i["frames"] for i in infos)
    duration = n_frames / fps

    if args.grid:
        with open(args.grid, "r", encoding="utf-8-sig") as f:
            grid = json.load(f)
        bpm, offset = float(grid["bpm"]), float(grid.get("offset_s", 0.0))
    else:
        bpm, offset = args.bpm, 0.0
    step = args.every * 60.0 / bpm

    # Schnittpunkte in Frames (aufs CFR-Raster gerundet), 0 und Ende erzwungen
    bounds = {0, n_frames}
    t = offset
    while t < duration:
        fr = round(t * fps)
        if 0 < fr < n_frames:
            bounds.add(fr)
        t += step
    bounds = sorted(bounds)
    segments = [(a, b, i % len(args.videos)) for i, (a, b) in
                enumerate(zip(bounds, bounds[1:]))]
    if len(segments) < 2:
        fail(f"nur {len(segments)} Segment(e) — Grid/BPM pruefen (step={step:.3f}s, "
             f"duration={duration:.3f}s)")

    print(f"Schnittliste ({len(segments)} Segmente @ {bpm:g} BPM, alle {args.every} Beats, "
          f"{fps:g} fps):")
    filt, concat_in = [], []
    for k, (a, b, vi) in enumerate(segments):
        print(f"  {a:3d}-{b:3d}  ({(b - a):2d} Frames)  {os.path.basename(args.videos[vi])}")
        filt.append(f"[{vi}:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[s{k}]")
        concat_in.append(f"[s{k}]")
    filt.append("".join(concat_in) + f"concat=n={len(segments)}:v=1:a=0[out]")

    cmd = [ffmpeg, "-y", "-v", "error"]
    for v in args.videos:
        cmd += ["-i", v]
    cmd += ["-filter_complex", ";".join(filt), "-map", "[out]",
            "-map_metadata", "-1", "-bitexact",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-r", str(ref["fps"]), "-movflags", "+faststart",
            args.out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        fail("ffmpeg-Schnitt fehlgeschlagen:\n" + r.stderr.decode("utf-8", "replace")[-600:], 3)

    got = probe(ffprobe, args.out)
    if got["frames"] != n_frames:
        fail(f"Postcondition: Output hat {got['frames']} Frames, erwartet {n_frames}.", 4)
    print(f"DONE: {args.out} ({got['frames']} Frames @ {fps:g} fps, "
          f"{got['frames'] / fps:.2f}s, stumm — Audio kommt im Export)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
