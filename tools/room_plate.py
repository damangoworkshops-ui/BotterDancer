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


def masked_plate(arr, mask_dir, min_samples=3, dump_mask=None):
    """Median NUR ueber Frames, in denen der Pixel unverdeckt ist.

    Der blinde Median scheitert, sobald jemand konstant an derselben Stelle
    steht (Formations-Choreo) — dann ist die Person an diesem Pixel in der
    Mehrzahl der Frames und gewinnt den Median. Mit den Personen-Masken faellt
    sie aus der Stichprobe; uebrig bleibt der echte Raum. Pixel, die NIE frei
    sind, fallen auf den blinden Median zurueck (besser als ein Loch).
    """
    import cv2
    import glob
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not files:
        print(f"FEHLER: keine Masken in {mask_dir}", file=sys.stderr)
        sys.exit(2)
    n, h, w = arr.shape[:3]
    # Masken und Videoframes muessen DENSELBEN Moment zeigen. Die Masken kommen
    # aus der Pose-Extraktion (typisch 16 fps resampled), das Rohvideo laeuft oft
    # mit 24/30 — dann ist Maske i ein anderer Zeitpunkt als Frame i und der
    # maskierte Median bringt nichts (Befund 06.08.: sah aus wie der blinde).
    if len(files) != n:
        print(f"FEHLER: {len(files)} Masken, aber {n} Videoframes — Maske i und "
              f"Frame i zeigen dann verschiedene Momente. Video mit derselben fps "
              f"laden, mit der die Posen extrahiert wurden (z.B. force_rate 16).",
              file=sys.stderr)
        sys.exit(2)
    masks = np.zeros((n, h, w), bool)
    for i in range(min(n, len(files))):
        m = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks[i] = m > 96
    free = ~masks                                   # True = Pixel unverdeckt
    counts = free.sum(axis=0)
    print(f"[plate] maskierter Median: Pixel im Mittel {counts.mean():.0f}/{n} frei, "
          f"{int((counts < min_samples).sum())} Pixel dauerhaft verdeckt")
    blind = np.median(arr, axis=0)
    out = blind.copy()
    ys, xs = np.where(counts >= min_samples)
    # spaltenweise, damit der Speicher nicht explodiert
    for y in np.unique(ys):
        sel = xs[ys == y]
        for x in sel:
            out[y, x] = np.median(arr[free[:, y, x], y, x], axis=0)
    out = out.astype(np.uint8)

    # Restflaechen, die NIE frei sind (bei fester Formation real 15%+ des Bildes),
    # kann kein Median fuellen — dort steht sonst weiter ein Geist. Telea-Inpainting
    # zieht die Umgebung hinein; bei glatten Studiowaenden reicht das, bei
    # strukturiertem Hintergrund bleibt es eine Naeherung.
    hopeless = (counts < min_samples).astype(np.uint8)
    if hopeless.any():
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        grown = cv2.dilate(hopeless, k)
        if dump_mask:
            # Fuers Modell-Inpainting: diese Flaechen kann kein Median fuellen.
            # Telea zieht nur die Umgebung hinein — bei strukturiertem
            # Hintergrund bleiben Schlieren, die ein Bildmodell besser loest.
            cv2.imwrite(dump_mask, (grown * 255).astype(np.uint8))
            print(f"[plate] Maske der aussichtslosen Flaechen -> {dump_mask}")
        out = cv2.inpaint(out, grown, 6, cv2.INPAINT_TELEA)
        print(f"[plate] {int(hopeless.sum())} dauerhaft verdeckte Pixel "
              f"({hopeless.mean() * 100:.1f}%) per Telea-Inpainting gefuellt")
    return out


def main():
    ap = argparse.ArgumentParser(description="Raum ohne Personen (zeitlicher Median)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=81, help="Laenge des Plates")
    ap.add_argument("--fps", type=float, help="fps des Plates (Default: wie die Quelle). "
                    "WICHTIG: muss zur fps der Figuren-Clips passen, sonst laeuft das "
                    "Composite zeitlich falsch (Befund 06.08.: 30er-Plate + 16er-Figuren "
                    "ergab einen um 1.9x zu schnellen Clip).")
    ap.add_argument("--sample", type=int, default=0,
                    help="wie viele Quellframes in den Median (0 = alle)")
    ap.add_argument("--masks", help="Verzeichnis mit Personen-Masken (pose_silhouette.py, "
                    "OHNE --track = alle Personen). Dann wird pro Pixel nur ueber die "
                    "Frames gemittelt, in denen dort NIEMAND steht — das entfernt die "
                    "Geister, die der blinde Median bei fester Formation stehen laesst.")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--dump-mask", help="PNG-Pfad: Maske der Flaechen, die kein Median "
                    "fuellen kann (Eingabe fuer Modell-Inpainting)")
    ap.add_argument("--dump-plate", help="PNG-Pfad: das Plate als Standbild")
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
    if args.masks:
        plate = masked_plate(arr, args.masks, dump_mask=args.dump_mask)
    else:
        plate = np.median(arr, axis=0).astype(np.uint8)
    h, w = plate.shape[:2]
    if args.fps:
        fps = args.fps

    # Restspuren pruefen: wo weicht der Median stark von der Mehrheit ab?
    dev = np.median(np.abs(arr.astype(np.float32) - plate), axis=0).mean()
    print(f"[plate] {len(frames)} Frames {w}x{h} @{fps:g}fps, "
          f"mittlere Restabweichung {dev:.1f} (klein = sauberer Raum)")
    if dev > 25:
        print("WARNUNG: hohe Restabweichung — Kamera bewegt sich oder Personen "
              "stehen zu lange still; Plate kann Geister enthalten.", file=sys.stderr)

    if args.dump_plate:
        cv2.imwrite(args.dump_plate, plate)
        print(f"[plate] Standbild -> {args.dump_plate}")
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
