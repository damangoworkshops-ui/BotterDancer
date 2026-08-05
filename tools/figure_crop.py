#!/usr/bin/env python
"""BotterDancer — Figuren-Crop: jede Taenzerin in voller Render-Aufloesung.

Das eigentliche Identitaets-Limit bei Gruppen ist nicht das Modell, sondern die
AUFLOESUNG PRO FIGUR. Gemessen am IZNA-Material: eine Taenzerin ist 280x110 px
gross — rund 5%% der Bildflaeche. Rendert man stattdessen einen Ausschnitt um
sie herum auf 512x768, bekommt dieselbe Figur das etwa 13-fache an Pixeln, und
die Referenz setzt sich entsprechend deutlicher durch.

Dieses Tool schneidet die Pose-Sequenz eines Tracks auf ein figurzentriertes
Fenster und schreibt die Transformation als Sidecar, damit crew_composite das
Renderergebnis spaeter exakt zurueckprojizieren kann.

  python figure_crop.py --pose-dir <track> --pose-json <pose.json> --track 0
                        --outdir <cropdir> [--width 512] [--height 768]

Fensterlogik: EINE feste Fenstergroesse fuer die ganze Sequenz (aus dem
groessten Frame plus Rand, aufs Zielseitenverhaeltnis gebracht) und ein
geglaettetes Zentrum — sonst pumpt die Figur in der Groesse und wackelt.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def kps(person):
    return np.array(person["pose_keypoints_2d"], dtype=float).reshape(-1, 3)


def boxes_from_pose_pngs(files, margin=0.35):
    """(F,4) Bounding-Boxen [x0,y0,x1,y1] direkt aus den gerenderten Skelett-PNGs.

    WICHTIG: nicht aus dem Pose-JSON ableiten. Die Track-PNGs stammen aus dem
    stabilen Tracking von crew_pose, eine Groessen-Sortierung im JSON trifft
    pro Frame womoeglich eine ANDERE Person — dann sitzt das Crop-Fenster auf
    der falschen Taenzerin (empirisch 06.08.: der Ausschnitt kam schwarz zurueck).
    """
    import cv2
    out = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            out.append(None)
            continue
        ys, xs = np.where(img.max(axis=2) > 20)
        if len(xs) < 10:
            out.append(None)
            continue
        w = float(xs.max() - xs.min())
        h = float(ys.max() - ys.min())
        out.append([xs.min() - margin * w, ys.min() - margin * h,
                    xs.max() + margin * w, ys.max() + margin * h])
    return out


def smooth(vals, win=9):
    if win <= 1 or len(vals) < 3:
        return np.asarray(vals, dtype=float)
    pad = win // 2
    x = np.pad(np.asarray(vals, dtype=float), pad, mode="edge")
    return np.convolve(x, np.ones(win) / win, mode="valid")


def plan_windows(boxes, canvas_w, canvas_h, out_w, out_h, smooth_win=9):
    """Feste Fenstergroesse + geglaettetes Zentrum -> (F,4) Fenster in Pixeln."""
    valid = [b for b in boxes if b is not None]
    if not valid:
        return None
    arr = np.array(valid)
    # Perzentil statt Maximum: EIN Frame mit voll gestreckten Armen wuerde das
    # Fenster fuer die ganze Sequenz aufblasen und den Aufloesungsgewinn
    # auffressen (gemessen 06.08.: max-basiert nur 1.8x). Die wenigen groesseren
    # Frames werden vom Rand aufgefangen, den margin ohnehin vorhaelt.
    need_w = float(np.percentile(arr[:, 2] - arr[:, 0], 92))
    need_h = float(np.percentile(arr[:, 3] - arr[:, 1], 92))
    ar = out_w / out_h
    # Fenster aufs Zielverhaeltnis aufblasen (nie beschneiden)
    win_h = max(need_h, need_w / ar)
    win_w = win_h * ar
    win_w = min(win_w, canvas_w)
    win_h = min(win_h, canvas_h)
    win_w = min(win_w, win_h * ar)
    win_h = win_w / ar

    cx, cy, last = [], [], None
    for b in boxes:
        if b is None:
            b = last if last is not None else valid[0]
        last = b
        cx.append((b[0] + b[2]) / 2.0)
        cy.append((b[1] + b[3]) / 2.0)
    cx = smooth(cx, smooth_win)
    cy = smooth(cy, smooth_win)
    cx = np.clip(cx, win_w / 2, canvas_w - win_w / 2)
    cy = np.clip(cy, win_h / 2, canvas_h - win_h / 2)
    return np.stack([cx - win_w / 2, cy - win_h / 2,
                     np.full(len(cx), win_w), np.full(len(cy), win_h)], axis=1)


def main():
    ap = argparse.ArgumentParser(description="Figuren-Crop fuer hoehere Aufloesung pro Figur")
    ap.add_argument("--pose-dir", required=True,
                    help="gerenderte Track-Pose-PNGs (crew_pose --split), genau EINE Figur")
    ap.add_argument("--canvas", nargs=2, type=int, metavar=("W", "H"),
                    help="Canvas-Groesse; Default: aus dem ersten PNG")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--margin", type=float, default=0.35)
    args = ap.parse_args()

    import cv2
    import glob
    from PIL import Image

    files = sorted(glob.glob(os.path.join(args.pose_dir, "*.png")))
    if not files:
        print(f"FEHLER: keine PNGs in {args.pose_dir}", file=sys.stderr)
        return 2
    if args.canvas:
        cw, ch = args.canvas
    else:
        probe = cv2.imread(files[0])
        ch, cw = probe.shape[:2]

    boxes = boxes_from_pose_pngs(files, args.margin)
    found = sum(1 for b in boxes if b is not None)
    wins = plan_windows(boxes, cw, ch, args.width, args.height)
    if wins is None:
        print(f"FEHLER: in keinem Frame ein Skelett gefunden.", file=sys.stderr)
        return 3
    gain = (args.width * args.height) / max(wins[0, 2] * wins[0, 3], 1.0)
    print(f"[crop] Figur in {found}/{len(files)} Frames erkannt; "
          f"Fenster {wins[0, 2]:.0f}x{wins[0, 3]:.0f} -> {args.width}x{args.height} "
          f"({gain:.1f}x mehr Pixel pro Figur)")

    tmp = args.outdir.rstrip("\\/") + f".tmp-{os.getpid()}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img.shape[1] != cw or img.shape[0] != ch:
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = wins[i]
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
        crop = img[max(0, y0):min(ch, y1), max(0, x0):min(cw, x1)]
        crop = cv2.resize(crop, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        Image.fromarray(crop[:, :, ::-1]).save(os.path.join(tmp, f"pose_{i:04d}.png"))

    if os.path.isdir(args.outdir):
        foreign = [e for e in os.listdir(args.outdir) if not e.startswith("pose_")]
        if foreign:
            print(f"FEHLER: {args.outdir} enthaelt fremde Eintraege {foreign[:3]}.",
                  file=sys.stderr)
            return 3
        shutil.rmtree(args.outdir)
    os.rename(tmp, args.outdir)

    meta = {"source_pose_dir": os.path.abspath(args.pose_dir),
            "canvas": [cw, ch], "out": [args.width, args.height],
            "windows": [[float(v) for v in row] for row in wins],
            "pixel_gain": float(gain)}
    with open(args.outdir.rstrip("\\/") + ".crop.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"DONE: {len(files)} Frames -> {args.outdir} (+ .crop.json fuer die Rueckprojektion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
