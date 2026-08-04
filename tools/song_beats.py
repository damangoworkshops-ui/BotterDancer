#!/usr/bin/env python
"""BotterDancer — echte Songs in die Beat-Lane: Beat This! + Segment-Wahl.

Analysiert einen Song (Beat This!, MIT, ohne --dbn — Research-Entscheidung
04.07.), waehlt ein choreo-langes Segment mit stabilem Tempo, das an einem
Downbeat beginnt, und schreibt:
  <name>_segment.wav        — Song-Ausschnitt (NUR fuer lokale Vorschau;
                              Produktregel: Original-Song verlaesst nie die
                              Maschine, Export bekommt rechtefreies Audio)
  <name>_segment.beats.json — Beat-Grid (motion_warp-kompatibel; enthaelt die
                              ECHTEN Beat-Zeiten inkl. Tempo-Wobble)

  python song_beats.py --audio song.mp3 --out-dir <dir> [--duration 6]
                       [--start 63.2]   (sonst automatische Segment-Wahl)

Laeuft im dedizierten Audio-venv (.venv-audio, CPU-Torch) — die beiden
Produktions-venvs (ComfyUI/GVHMR) bleiben unangetastet.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

FFMPEG_FALLBACK = (r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages"
                   r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                   r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe")


def pick_segment(beats, downbeats, duration, seg_len):
    """Startzeit: Downbeat im mittleren Songdrittel-bis-80%, dessen Fenster die
    stabilsten Beat-Intervalle hat (Chorus-artige Passagen sind stabil)."""
    beats = np.asarray(beats, dtype=float)
    cands = [d for d in np.asarray(downbeats, dtype=float)
             if 0.2 * duration <= d <= 0.8 * duration - seg_len]
    if not cands:
        cands = [d for d in downbeats if d <= duration - seg_len] or [beats[0]]
    best = None
    for d in cands:
        w = beats[(beats >= d) & (beats <= d + seg_len)]
        if len(w) < 4:
            continue
        iv = np.diff(w)
        score = float(np.std(iv))
        if best is None or score < best[0]:
            best = (score, float(d))
    return best[1] if best else float(cands[0])


def main():
    ap = argparse.ArgumentParser(description="Song-Beat-Analyse + Segment-Extraktion")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--duration", type=float, default=6.0, help="Segmentlaenge (s)")
    ap.add_argument("--start", type=float, help="Segmentstart erzwingen (sonst Auto-Wahl)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"FEHLER: {audio} existiert nicht.", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from beat_this.inference import File2Beats
    print(f"[song] Beat This! analysiert {audio.name} (CPU) ...")
    f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = f2b(str(audio))
    beats = np.asarray(beats, dtype=float)
    ivs = np.diff(beats)
    bpm_global = 60.0 / float(np.median(ivs))
    dur = float(beats[-1])
    print(f"[song] {len(beats)} Beats, {len(downbeats)} Downbeats, "
          f"~{bpm_global:.1f} BPM global")

    start = args.start if args.start is not None else pick_segment(
        beats, downbeats, dur, args.duration)
    seg_beats = beats[(beats >= start) & (beats <= start + args.duration)] - start
    seg_iv = np.diff(seg_beats)
    bpm_seg = 60.0 / float(np.median(seg_iv))
    print(f"[song] Segment: {start:.2f}s - {start + args.duration:.2f}s, "
          f"{len(seg_beats)} Beats, {bpm_seg:.1f} BPM, "
          f"Intervall-Streuung {float(np.std(seg_iv)) * 1000:.0f}ms")

    stem = audio.stem[:40].replace(" ", "_")
    wav = out_dir / f"{stem}_segment.wav"
    import shutil
    ffmpeg = shutil.which("ffmpeg") or FFMPEG_FALLBACK
    subprocess.run([ffmpeg, "-v", "error", "-ss", f"{start:.3f}", "-t",
                    f"{args.duration:.3f}", "-i", str(audio), "-ar", "44100",
                    "-ac", "2", "-y", str(wav)], check=True, capture_output=True)

    grid = {
        "bpm": round(bpm_seg, 3),
        "offset_s": round(float(seg_beats[0]), 4) if len(seg_beats) else 0.0,
        "duration_s": args.duration,
        "beats": [round(float(b), 4) for b in seg_beats],
        "source": f"Beat This! (dbn=False) auf {audio.name}, Segment {start:.2f}s"
                  f"+{args.duration:.0f}s — NUR lokale Analyse/Vorschau, Song wird nie exportiert",
    }
    grid_path = Path(str(wav) + ".beats.json")
    with open(grid_path, "w") as f:
        json.dump(grid, f, indent=1)
    print(f"DONE: {wav.name} + {grid_path.name} in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
