#!/usr/bin/env python
"""BotterDancer — Crew-Compositing: mehrere Solo-Renders zu einer Gruppe.

Warum nicht per background_video-Layering (der naheliegende Weg)? Empirisch
06.08.: Wan-Animate re-generiert die Szene bei jeder Ebene neu, dabei driften
die Farben kumulativ — Ebene 1 sauber, Ebene 2 cyan, Ebene 3 kollabiert.
Deshalb: jede Figur SOLO vor Studio rendern (farblich stabil), danach hier
zusammensetzen.

Freistellung ohne Greenscreen: bei statischer Kamera ist der zeitliche MEDIAN
jedes Pixels der Hintergrund — die Taenzerin bewegt sich, das Studio nicht.
Maske = Abweichung vom Median, morphologisch geglaettet und weichgezeichnet
(weiche Kante statt Treppenrand). Reihenfolge der Ebenen = Argument-Reihenfolge;
bei Ueberlappung gewinnt die spaetere (Vordergrund zuletzt angeben).

  python crew_composite.py --base solo_a.mp4 --overlay solo_b.mp4 solo_c.mp4
                           --out crew.mp4 [--thresh 18] [--feather 3]
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


def read_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"FEHLER: {path} nicht lesbar.", file=sys.stderr)
        sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.array(frames), float(fps)


def pose_boxes(pose_dir, shape, margin=0.10):
    """Pro Frame eine grosszuegige Bounding-Box der gezeichneten Pose (Skelett-PNGs
    sind schwarz mit farbigen Gliedmassen). Begrenzt die Maske auf DIESE Taenzerin —
    ohne das schleppt die Median-Maske Bodenschatten und Nachbarfiguren mit
    (empirisch 06.08.: 17% Maskenflaeche statt ~6%)."""
    import cv2
    import glob
    files = sorted(glob.glob(os.path.join(pose_dir, "*.png")))
    n, h, w = shape
    boxes = []
    for i in range(n):
        if i >= len(files):
            boxes.append(None)
            continue
        img = cv2.imread(files[i])
        if img is None:
            boxes.append(None)
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(img.max(axis=2) > 20)
        if len(xs) < 10:
            boxes.append(None)
            continue
        mx, my = int(margin * w), int(margin * h)
        boxes.append((max(0, xs.min() - mx), min(w, xs.max() + mx),
                      max(0, ys.min() - my), min(h, ys.max() + my)))
    return boxes


def person_mask(frames, thresh, feather, min_area=200, boxes=None):
    """(F,H,W) float-Maske 0..1 der bewegten Figur (statische Kamera vorausgesetzt)."""
    import cv2
    bg = np.median(frames, axis=0)                      # zeitlicher Median = Studio
    masks = np.empty(frames.shape[:3], dtype=np.float32)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i, f in enumerate(frames):
        d = np.linalg.norm(f.astype(np.float32) - bg, axis=2)
        m = (d > thresh).astype(np.uint8)
        if boxes is not None and i < len(boxes) and boxes[i] is not None:
            x0, x1, y0, y1 = boxes[i]
            keep = np.zeros_like(m)
            keep[y0:y1, x0:x1] = 1
            m = m * keep
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)     # Rauschen weg
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)   # Loecher im Koerper zu
        # nur die groesste zusammenhaengende Flaeche behalten (Rest = Artefakt)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[biggest, cv2.CC_STAT_AREA] >= min_area:
                m = (lab == biggest).astype(np.uint8)
            else:
                m[:] = 0
        if feather > 0:
            k = 2 * feather + 1
            masks[i] = cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)
        else:
            masks[i] = m.astype(np.float32)
    return np.clip(masks, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Solo-Renders zu einer Gruppe compositen")
    ap.add_argument("--base", required=True, help="Basis-Clip (liefert Hintergrund)")
    ap.add_argument("--overlay", nargs="+", required=True,
                    help="weitere Solo-Clips, spaetere gewinnen bei Ueberlappung")
    ap.add_argument("--overlay-pose", nargs="*", default=[],
                    help="je Overlay das zugehoerige Track-Pose-Verzeichnis — "
                         "begrenzt die Maske auf DIESE Taenzerin (empfohlen)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=18.0, help="Maskenschwelle (Farbdistanz)")
    ap.add_argument("--feather", type=int, default=3, help="weiche Kante in Pixeln")
    ap.add_argument("--no-match", action="store_true",
                    help="Hintergrund-Helligkeitsangleich der Overlays abschalten")
    ap.add_argument("--crf", type=int, default=12)
    args = ap.parse_args()

    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        print("FEHLER: ffmpeg nicht gefunden.", file=sys.stderr)
        return 2

    base, fps = read_frames(args.base)
    n, h, w = base.shape[:3]
    print(f"[comp] Basis {os.path.basename(args.base)}: {n} Frames {w}x{h} @{fps:g}fps")
    out = base.astype(np.float32)

    if args.overlay_pose and len(args.overlay_pose) != len(args.overlay):
        print(f"FEHLER: --overlay-pose braucht genau {len(args.overlay)} Eintraege.",
              file=sys.stderr)
        return 2
    for oi, ov_path in enumerate(args.overlay):
        ov, _ = read_frames(ov_path)
        if ov.shape[1:3] != (h, w):
            print(f"FEHLER: {ov_path} hat {ov.shape[2]}x{ov.shape[1]}, "
                  f"Basis {w}x{h}.", file=sys.stderr)
            return 2
        k = min(n, len(ov))
        boxes = (pose_boxes(args.overlay_pose[oi], (k, h, w))
                 if args.overlay_pose else None)
        m = person_mask(ov[:k], args.thresh, args.feather, boxes=boxes)
        if not args.no_match:
            # Hintergrund-Angleich: jeder Solo-Render generiert sein Studio leicht
            # anders hell. Ohne Korrektur zeichnet die weiche Maskenkante einen
            # sichtbaren Halo um jede eingefuegte Figur (empirisch 06.08.).
            base_bg = np.median(base.reshape(-1, 3), axis=0)
            ov_bg = np.median(ov[:k].reshape(-1, 3), axis=0)
            shift = base_bg - ov_bg
            ov = np.clip(ov.astype(np.float32) + shift, 0, 255)
            print(f"       Hintergrund-Angleich BGR {np.round(shift, 1)}")
        cover = float(m.mean())
        print(f"[comp] + {os.path.basename(ov_path)}: {k} Frames, "
              f"Maskenflaeche {cover * 100:.1f}%")
        if cover < 0.002:
            print(f"WARNUNG: Maske fast leer — --thresh senken?", file=sys.stderr)
        m3 = m[..., None]
        out[:k] = out[:k] * (1.0 - m3) + ov[:k].astype(np.float32) * m3

    frames = np.clip(out, 0, 255).astype(np.uint8)
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-map_metadata", "-1", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        print("FEHLER: ffmpeg:\n" + p.stderr.read().decode("utf-8", "replace")[-500:],
              file=sys.stderr)
        return 3
    print(f"DONE: {args.out} ({n} Frames, {1 + len(args.overlay)} Figuren)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
