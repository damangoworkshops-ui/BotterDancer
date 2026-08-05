#!/usr/bin/env python
"""BotterDancer — Naht-Pruefung fuer verkettete Renders (lange Routinen).

Beim Chunk-Chaining (make_chain_workflow.py) ist die Frage nicht "sieht gut
aus", sondern: springt es an der Chunk-Grenze? Dieses Tool misst das objektiv
und vergleicht IMMER gegen die lokale Baseline derselben Sequenz — absolute
Schwellen waeren bei tanzender Bewegung sinnlos, weil schnelle Moves von Natur
aus grosse Frame-Differenzen erzeugen.

Zwei Kennzahlen pro Naht:
- Bewegungssprung: Frame-Differenz an der Naht / Median der Nachbar-Differenzen
  (Ratio ~1 = Naht nicht von normaler Bewegung unterscheidbar = gut).
- Identitaets-/Farbsprung: Bhattacharyya-Distanz der Figur-Histogramme der
  Fenster vor/nach der Naht, ebenfalls gegen eine Baseline aus verschobenen
  Fenstern derselben Sequenz normiert.

  python seam_check.py --video chain.mp4 --seams 81
  python seam_check.py --video chain.mp4 --chunk-length 81 --chunks 3

Exit 0 = alle Naehte unauffaellig, 4 = mindestens eine Naht ueber Schwelle.
"""
import argparse
import json
import os
import sys

import numpy as np


def load_gray_and_masks(path, max_frames=4096):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"FEHLER: {path} nicht lesbar.", file=sys.stderr)
        sys.exit(2)
    grays, hists = [], []
    while len(grays) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (128, 192), interpolation=cv2.INTER_AREA)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
        # Figur-Maske ueber Ecken-Hintergrund (Studio-BG ist nahezu uniform)
        corners = np.concatenate([small[:6, :6].reshape(-1, 3), small[:6, -6:].reshape(-1, 3),
                                  small[-6:, :6].reshape(-1, 3), small[-6:, -6:].reshape(-1, 3)])
        bg = np.median(corners, axis=0)
        mask = (np.linalg.norm(small.astype(np.int16) - bg, axis=2) > 28).astype(np.uint8)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
        cv2.normalize(h, h)
        hists.append(h)
    cap.release()
    return np.array(grays), hists


def window_hist(hists, lo, hi):
    import cv2
    w = hists[max(0, lo):hi]
    if not w:
        return None
    h = sum(w) / len(w)
    cv2.normalize(h, h)
    return h


def motion_ratio_at(diffs, seam, win=10):
    """(ratio, seam_diff, local_baseline, perzentil) fuer einen Uebergang."""
    i = seam - 1
    lo, hi = max(0, i - win), min(len(diffs), i + win + 1)
    neigh = np.concatenate([diffs[lo:i], diffs[i + 1:hi]])
    base = float(np.median(neigh)) if len(neigh) else float("nan")
    ratio = float(diffs[i] / base) if base > 1e-6 else float("nan")
    pct = float((diffs < diffs[i]).mean() * 100)
    return ratio, float(diffs[i]), base, pct


def check_seam(grays, hists, seam, win=10, pose_diffs=None):
    """(motion_ratio, color_ratio, details) fuer eine Naht bei Index `seam`
    (= erster Frame des neuen Chunks).

    pose_diffs: optionale Differenzkurve der POSE-Eingabesequenz. Damit wird
    zwischen Motion-Ursache und Render-Ursache unterschieden — der wichtigste
    Kontrollversuch beim Kettenrendern: zeigt schon der Input an dieser Stelle
    einen Sprung (Richtungswechsel, harter Move), ist die Naht unschuldig.
    Empirisch 06.08. am ersten echten Chain-Render: Render 2.53x, Input 1.38x
    -> der Ueberschuss ist das eigentliche Chunk-Artefakt."""
    import cv2
    diffs = np.abs(np.diff(grays, axis=0)).mean(axis=(1, 2))  # (F-1,)
    i = seam - 1  # Uebergang seam-1 -> seam
    motion_ratio, seam_diff, base, pct = motion_ratio_at(diffs, seam, win)
    pose_ratio = None
    if pose_diffs is not None and seam - 1 < len(pose_diffs):
        pose_ratio = motion_ratio_at(pose_diffs, seam, win)[0]

    hb, ha = window_hist(hists, seam - win, seam), window_hist(hists, seam, seam + win)
    color = float(cv2.compareHist(hb, ha, cv2.HISTCMP_BHATTACHARYYA)) if hb is not None and ha is not None else float("nan")
    # Baseline: gleich grosse Fensterpaare ohne Naht dazwischen
    refs = []
    for off in (-3 * win, -2 * win, 2 * win, 3 * win):
        c = seam + off
        a, b = window_hist(hists, c - win, c), window_hist(hists, c, c + win)
        if a is not None and b is not None and 0 <= c - win and c + win <= len(hists):
            refs.append(float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)))
    color_base = float(np.median(refs)) if refs else float("nan")
    color_ratio = color / color_base if color_base and color_base > 1e-6 else float("nan")
    # Ueberschuss ueber den Input: das ist der Anteil, den das RENDERN erzeugt.
    excess = (motion_ratio / pose_ratio) if (pose_ratio and pose_ratio > 1e-6) else None
    return motion_ratio, color_ratio, {
        "diff_seam": float(diffs[i]), "diff_base": base,
        "seam_percentile": pct,  # wo liegt die Naht-Differenz in der Sequenz?
        "pose_motion_ratio": pose_ratio, "render_excess_ratio": excess,
        "color_seam": color, "color_base": color_base}


def main():
    ap = argparse.ArgumentParser(description="Naht-Pruefung fuer Chunk-Ketten")
    ap.add_argument("--video", required=True)
    ap.add_argument("--seams", type=int, nargs="*", help="Frame-Indizes der Chunk-Starts")
    ap.add_argument("--chunk-length", type=int, help="alternativ: gleichmaessige Chunks")
    ap.add_argument("--chunks", type=int, help="Anzahl Chunks (mit --chunk-length)")
    ap.add_argument("--pose-dir", help="Pose-PNG-Sequenz des Renders — Kontrollmessung: "
                    "trennt Motion-Ursache (Input springt auch) von Render-Artefakt")
    ap.add_argument("--max-motion-ratio", type=float, default=3.5,
                    help="Naht-Differenz / lokale Baseline. Default 3.5 aus dem ersten "
                         "echten Chain-Render kalibriert: dort war 2.53x visuell "
                         "unauffaellig (86. Perzentil der Sequenz).")
    ap.add_argument("--max-excess-ratio", type=float, default=2.5,
                    help="mit --pose-dir: Render-Ratio / Input-Ratio (der Anteil, den "
                         "erst das Rendern erzeugt) — die aussagekraeftigere Zahl")
    ap.add_argument("--max-color-ratio", type=float, default=3.0)
    ap.add_argument("--json-out", help="Report zusaetzlich als JSON")
    args = ap.parse_args()

    seams = list(args.seams or [])
    if args.chunk_length and args.chunks:
        seams += [args.chunk_length * k for k in range(1, args.chunks)]
    seams = sorted(set(s for s in seams if s > 0))
    if not seams:
        print("FEHLER: --seams oder --chunk-length/--chunks angeben.", file=sys.stderr)
        return 2

    grays, hists = load_gray_and_masks(args.video)
    pose_diffs = None
    if args.pose_dir:
        import cv2
        import glob
        files = sorted(glob.glob(os.path.join(args.pose_dir, "*.png")))
        if files:
            pg = np.array([cv2.cvtColor(cv2.resize(cv2.imread(f), (128, 192)),
                                        cv2.COLOR_BGR2GRAY).astype(np.float32) for f in files])
            pose_diffs = np.abs(np.diff(pg, axis=0)).mean(axis=(1, 2))
            print(f"[seam] Pose-Kontrolle: {len(files)} Input-Frames")
        else:
            print(f"WARNUNG: keine PNGs in {args.pose_dir}", file=sys.stderr)

    print(f"[seam] {os.path.basename(args.video)}: {len(grays)} Frames, "
          f"Naehte bei {seams}")
    report, bad = [], []
    for s in seams:
        if not (1 <= s < len(grays)):
            print(f"  Naht {s}: ausserhalb der Sequenz — uebersprungen", file=sys.stderr)
            continue
        mr, cr, det = check_seam(grays, hists, s, pose_diffs=pose_diffs)
        excess = det.get("render_excess_ratio")
        flag = ""
        # Mit Pose-Kontrolle entscheidet der Ueberschuss, sonst die Rohratio.
        if excess is not None:
            if excess > args.max_excess_ratio:
                flag += "  <-- RENDER-NAHT-ARTEFAKT"
        elif mr > args.max_motion_ratio:
            flag += "  <-- BEWEGUNGSSPRUNG"
        if cr > args.max_color_ratio:
            flag += "  <-- FARB-/IDENTITAETSSPRUNG"
        if flag:
            bad.append(s)
        line = (f"  Naht @{s:4d}: Bewegung {mr:.2f}x Baseline "
                f"(Perzentil {det['seam_percentile']:.0f} der Sequenz), "
                f"Farbe {cr:.2f}x")
        if excess is not None:
            line += f", Input {det['pose_motion_ratio']:.2f}x -> Render-Ueberschuss {excess:.2f}x"
        print(line + flag)
        report.append({"seam": s, "motion_ratio": mr, "color_ratio": cr, **det})

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"video": os.path.abspath(args.video), "seams": report,
                       "thresholds": {"motion": args.max_motion_ratio,
                                      "color": args.max_color_ratio}}, f, indent=1)
    if bad:
        print(f"ERGEBNIS: {len(bad)} auffaellige Naht/Naehte bei {bad}", file=sys.stderr)
        return 4
    print("ERGEBNIS: alle Naehte unauffaellig (nicht von normaler Bewegung unterscheidbar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
