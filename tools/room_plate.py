#!/usr/bin/env python
"""BotterDancer — Raum-Plate: der echte Raum OHNE Taenzer:innen (Room-Keeper).

Warum das noetig ist (empirisch 06.08.): Wan-Animate mit background_video +
character_mask ersetzt die Figur NICHT, wenn im Hintergrundvideo an dieser
Stelle noch die Original-Person steht — das Modell uebernimmt sie einfach.
Kontrolltest: mit Vollmaske wird ersetzt, mit Silhouetten-Maske nicht.
Die Maske sagt nur WO neu erzeugt werden darf; sie loescht die Vorlage nicht.

Loesung: ein Plate, in dem der Raum steht und die Personen weg sind. Bei
statischer Kamera liefert der zeitliche MEDIAN genau das — jede Stelle des
Bodens ist in der Mehrzahl der Frames unverdeckt (dieselbe Beobachtung wie
in crew_composite, nur andersherum genutzt).

  python room_plate.py --video quelle.mp4 --out plate.mp4 [--frames 81]

Ergebnis ist ein Standbild-Video (konstantes Plate) in Laenge/fps der Quelle.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

FFMPEG_FALLBACK_DIR = (r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages"
                       r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                       r"\ffmpeg-8.1-full_build\bin")


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(FFMPEG_FALLBACK_DIR, name + ".exe")
    return cand if os.path.isfile(cand) else None


def main():
    ap = argparse.ArgumentParser(description="Raum ohne Personen (zeitlicher Median)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=81, help="Laenge des Plates")
    ap.add_argument("--sample", type=int, default=0,
                    help="wie viele Quellframes in den Median (0 = alle)")
    ap.add_argument("--crf", type=int, default=12)
    args = ap.parse_args()

    import cv2
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        print("FEHLER: ffmpeg nicht gefunden.", file=sys.stderr)
        return 2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FEHLER: {args.video} nicht lesbar.", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 8:
        print(f"FEHLER: nur {len(frames)} Frames — Median braucht mehr.", file=sys.stderr)
        return 3
    arr = np.array(frames)
    if args.sample and args.sample < len(arr):
        idx = np.linspace(0, len(arr) - 1, args.sample).astype(int)
        arr = arr[idx]
    plate = np.median(arr, axis=0).astype(np.uint8)
    h, w = plate.shape[:2]

    # Restspuren pruefen: wo weicht der Median stark von der Mehrheit ab?
    dev = np.median(np.abs(arr.astype(np.float32) - plate), axis=0).mean()
    print(f"[plate] {len(frames)} Frames {w}x{h} @{fps:g}fps, "
          f"mittlere Restabweichung {dev:.1f} (klein = sauberer Raum)")
    if dev > 25:
        print("WARNUNG: hohe Restabweichung — Kamera bewegt sich oder Personen "
              "stehen zu lange still; Plate kann Geister enthalten.", file=sys.stderr)

    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(np.repeat(plate[None], args.frames, axis=0).tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        print("FEHLER ffmpeg:\n" + p.stderr.read().decode("utf-8", "replace")[-400:],
              file=sys.stderr)
        return 3
    print(f"DONE: {args.out} ({args.frames} Frames konstantes Raum-Plate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
