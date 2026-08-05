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
import json
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


def _spatial_mask(frames, thresh, feather, min_area=200):
    """Maske ueber die RANDFARBE jedes Frames — fuer figurzentrierte Crops, wo
    der zeitliche Median die Figur selbst ist. Der Studio-Hintergrund ist am
    Bildrand praktisch immer unverdeckt."""
    import cv2
    n, h, w = frames.shape[:3]
    masks = np.empty((n, h, w), dtype=np.float32)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band = max(4, w // 40)
    for i, f in enumerate(frames):
        edge = np.concatenate([f[:, :band].reshape(-1, 3), f[:, -band:].reshape(-1, 3),
                               f[:band, :].reshape(-1, 3)])
        bg = np.median(edge, axis=0)
        d = np.linalg.norm(f.astype(np.float32) - bg, axis=2)
        m = (d > thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if cnt > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = ((lab == big).astype(np.uint8) if stats[big, cv2.CC_STAT_AREA] >= min_area
                 else np.zeros_like(m))
        if feather > 0:
            kk = 2 * feather + 1
            masks[i] = cv2.GaussianBlur(m.astype(np.float32), (kk, kk), 0)
        else:
            masks[i] = m.astype(np.float32)
    return np.clip(masks, 0.0, 1.0)


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


def person_mask(frames, thresh, feather, min_area=200, boxes=None, spatial_bg=False):
    """(F,H,W) float-Maske 0..1 der Figur.

    Standard: zeitlicher Median als Hintergrund (statische Kamera, Figur bewegt
    sich durchs Bild). Das versagt bei FIGURZENTRIERTEN CROPS — dort steht die
    Figur konstant in der Mitte, der Median IST die Figur und die Maske wird
    unbrauchbar (empirisch 06.08.: Studio-Reste im Composite als Schlieren).
    spatial_bg=True nimmt stattdessen die Randfarbe JEDES Frames als Referenz.
    """
    import cv2
    if spatial_bg:
        return _spatial_mask(frames, thresh, feather, min_area)
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


WAN_SCALE_BIAS = 1.36
"""Wan rendert die Figur groesser als die Pose vorgibt — auch im Vollbild.
Gemessen 06.08. am IZNA-Material: Skelett 71%% der Bildhoehe, gerenderte Figur
97%%. Das ist keine Stoerung, sondern Teil des Looks: ALLE Figuren erscheinen
so. Eine Crop-Figur muss deshalb auf DIESE Groesse normiert werden, nicht auf
die nackte Pose-Groesse — sonst steht sie zu klein neben ihren Nachbarinnen
(erst gemessen 0.58 statt der passenden ~0.75).
"""


def scale_correction(frames, crop_meta, full_pose_dir=None, sample=7,
                     bias=WAN_SCALE_BIAS):
    """Wie stark muss der zurueckprojizierte Ausschnitt verkleinert werden?

    Zielgroesse = Skeletthoehe im Vollbild * WAN_SCALE_BIAS (also so gross, wie
    ein normaler Vollbild-Render dieselbe Figur gemacht haette). Istgroesse =
    Figurhoehe im Crop, auf das Fenster zurueckgerechnet. Ohne Vollbild-Posen
    faellt die Funktion auf eine Verhaeltnis-Schaetzung zurueck.
    """
    import cv2
    import glob
    if full_pose_dir and os.path.isdir(full_pose_dir):
        full_files = sorted(glob.glob(os.path.join(full_pose_dir, "*.png")))
        wins = crop_meta["windows"]
        _, oh = crop_meta["out"]
        ratios = []
        n = min(len(frames), len(full_files), len(wins))
        for i in np.linspace(0, n - 1, min(sample * 2, n)).astype(int):
            f = frames[int(i)]
            bg = np.median(np.concatenate([f[:8, :8].reshape(-1, 3),
                                           f[:8, -8:].reshape(-1, 3)]), axis=0)
            m = np.linalg.norm(f.astype(np.int16) - bg, axis=2) > 40
            ys, _ = np.where(m)
            fp = cv2.imread(full_files[int(i)])
            if len(ys) < 50 or fp is None:
                continue
            pys, _ = np.where(fp.max(axis=2) > 20)
            if len(pys) < 20:
                continue
            # Soll = so gross, wie ein Vollbild-Render diese Figur gemacht haette
            soll = float(pys.max() - pys.min()) * bias
            ist = float(ys.max() - ys.min()) * wins[int(i)][3] / oh
            if ist > 5 and soll > 5:
                ratios.append(soll / ist)
        if ratios:
            return float(np.clip(np.median(ratios), 0.4, 1.2))
    return _scale_correction_ratio(frames, crop_meta, sample)


def _scale_correction_ratio(frames, crop_meta, sample=7):
    """Wan rendert die Figur groesser als die Pose vorgibt — es orientiert sich
    an der (formatfuellenden) Referenz. Gemessen 06.08. am Crop: Skelett 71%
    der Bildhoehe, gerenderte Figur 97%. Ohne Korrektur landet die Figur beim
    Zurueckprojizieren rund 1.4x zu gross in der Szene.

    Korrektur: gerenderte Figurhoehe gegen die Skeletthoehe derselben Frames
    messen und den Ausschnitt entsprechend verkleinern (um den Figurfuss herum,
    denn der Boden ist der Bezug — sonst schwebt oder versinkt die Figur).
    """
    import cv2
    import glob
    pose_files = sorted(glob.glob(os.path.join(crop_meta["source_pose_dir"], "*.png")))
    ow, oh = crop_meta["out"]
    # Paarweise je Frame messen und nur die gestrecktesten Posen werten: dort
    # ist die Figur am ehesten formatfuellend und das Verhaeltnis stabil. In
    # gebueckten Frames ist das Skelett flach, waehrend das Modell die Figur
    # gross haelt — solche Frames verzerren den Wert nach unten (gemessen 0.53
    # statt der korrekten ~0.73). Maxima aus VERSCHIEDENEN Frames zu teilen ist
    # ebenfalls falsch, deshalb strikt paarweise.
    pairs = []
    n = min(len(frames), len(pose_files))
    idxs = np.linspace(0, n - 1, min(sample * 2, n)).astype(int)
    for i in idxs:
        f = frames[int(i)]
        bg = np.median(np.concatenate([f[:8, :8].reshape(-1, 3),
                                       f[:8, -8:].reshape(-1, 3)]), axis=0)
        m = np.linalg.norm(f.astype(np.int16) - bg, axis=2) > 40
        ys, _ = np.where(m)
        p = cv2.imread(pose_files[int(i)])
        if len(ys) < 50 or p is None:
            continue
        if p.shape[0] != oh or p.shape[1] != ow:
            p = cv2.resize(p, (ow, oh), interpolation=cv2.INTER_NEAREST)
        pys, _ = np.where(p.max(axis=2) > 20)
        if len(pys) < 20:
            continue
        fig_h = float(ys.max() - ys.min())
        pose_h = float(pys.max() - pys.min())
        if fig_h > 10 and pose_h > 10:
            pairs.append((pose_h, pose_h / fig_h))
    if not pairs:
        return 1.0
    pairs.sort(reverse=True)                      # gestreckteste Posen zuerst
    top = [r for _, r in pairs[:max(3, len(pairs) // 3)]]
    return float(np.clip(np.median(top), 0.5, 1.0))


def uncrop(frames, crop_meta, canvas_w, canvas_h, scale=1.0):
    """Figurzentrierte Crops zurueck auf die volle Leinwand projizieren.
    Ausserhalb des Fensters bleibt das Bild leer — die Maske entsteht ohnehin
    nur aus dem eingesetzten Bereich. `scale` < 1 verkleinert den Ausschnitt
    fussbuendig (Boden als Bezug)."""
    import cv2
    wins = crop_meta["windows"]
    out = np.zeros((len(frames), canvas_h, canvas_w, 3), np.uint8)
    for i, f in enumerate(frames):
        if i >= len(wins):
            break
        x, y, w, h = wins[i]
        tw, th = w * scale, h * scale
        # fussbuendig: untere Kante bleibt, Breite mittig
        x0 = int(round(x + (w - tw) / 2.0))
        y0 = int(round(y + (h - th)))
        x1, y1 = min(canvas_w, x0 + int(round(tw))), min(canvas_h, y0 + int(round(th)))
        x0, y0 = max(0, x0), max(0, y0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out[i, y0:y1, x0:x1] = cv2.resize(f, (x1 - x0, y1 - y0),
                                          interpolation=cv2.INTER_LANCZOS4)
    return out


def main():
    ap = argparse.ArgumentParser(description="Solo-Renders zu einer Gruppe compositen")
    ap.add_argument("--base", required=True, help="Basis-Clip (liefert Hintergrund)")
    ap.add_argument("--overlay", nargs="+", required=True,
                    help="weitere Solo-Clips, spaetere gewinnen bei Ueberlappung")
    ap.add_argument("--overlay-pose", nargs="*", default=[],
                    help="je Overlay das zugehoerige Track-Pose-Verzeichnis — "
                         "begrenzt die Maske auf DIESE Taenzerin (empfohlen)")
    ap.add_argument("--overlay-crop", nargs="*", default=[],
                    help="je Overlay ein .crop.json (figure_crop.py): der Clip ist dann "
                         "ein figurzentrierter Ausschnitt in hoher Aufloesung und wird "
                         "beim Mischen an seine Originalposition zurueckprojiziert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=18.0, help="Maskenschwelle (Farbdistanz)")
    ap.add_argument("--feather", type=int, default=3, help="weiche Kante in Pixeln")
    ap.add_argument("--no-match", action="store_true",
                    help="Hintergrund-Helligkeitsangleich der Overlays abschalten")
    ap.add_argument("--spatial-mask", action="store_true",
                    help="Maske bei Crops ueber die Randfarbe statt den zeitlichen "
                         "Median (bei uniformem Hintergrund manchmal besser)")
    ap.add_argument("--no-scale-fix", action="store_true",
                    help="Massstabs-Korrektur bei --overlay-crop abschalten (Wan rendert "
                         "die Figur groesser als die Pose vorgibt)")
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
    if args.overlay_crop and len(args.overlay_crop) != len(args.overlay):
        print(f"FEHLER: --overlay-crop braucht genau {len(args.overlay)} Eintraege.",
              file=sys.stderr)
        return 2
    for oi, ov_path in enumerate(args.overlay):
        ov, ov_fps = read_frames(ov_path)
        crop_mask = None
        if args.overlay_crop:
            entry = args.overlay_crop[oi]
            if entry and entry.lower() != "none":
                with open(entry, "r", encoding="utf-8-sig") as f:
                    cm = json.load(f)
                if cm["canvas"] != [w, h]:
                    print(f"FEHLER: {entry} gehoert zu Canvas {cm['canvas']}, "
                          f"Basis ist {[w, h]}.", file=sys.stderr)
                    return 2
                full_pose = (args.overlay_pose[oi] if args.overlay_pose else None)
                sc = (1.0 if args.no_scale_fix
                      else scale_correction(ov, cm, full_pose))
                # Maske VOR dem Uncrop bestimmen: im Crop findet der zeitliche
                # Median den Studio-Hintergrund korrekt. Nach dem Uncrop ist
                # ausserhalb des Fensters alles schwarz — der helle Crop-Kasten
                # wuerde dann komplett als "Figur" maskiert (Befund 06.08.:
                # sichtbarer Rahmen um die Figur).
                # Median-Maske trotz zentrierter Figur: der raeumliche Weg
                # (Randfarbe) erfasst im nicht-uniformen Studio zu viel
                # (gemessen 06.08.: 12-16%% statt 6-8%% Maskenflaeche, sichtbare
                # Kaesten). Beide Wege sind Naeherungen — der Median gewinnt.
                k = min(n, len(ov))
                mc = person_mask(ov[:k], args.thresh, args.feather,
                                 spatial_bg=args.spatial_mask)
                ov = uncrop(ov, cm, w, h, sc)
                crop_mask = uncrop(np.repeat((mc * 255).astype(np.uint8)[..., None], 3,
                                             axis=3), cm, w, h, sc)[..., 0] / 255.0
                print(f"       zurueckprojiziert aus {os.path.basename(entry)} "
                      f"({cm.get('pixel_gain', 0):.1f}x Aufloesungsgewinn, "
                      f"Massstabs-Korrektur {sc:.2f})")
        # fps-Konsistenz ist Pflicht: das Ergebnis erbt die fps der BASIS, ein
        # abweichender Overlay laeuft dann zeitlich falsch (Befund 06.08.:
        # 30er-Raum-Plate + 16er-Figuren = 1.9x zu schneller Clip; erst das
        # RIFE-Preflight-Gate hat es bemerkt).
        if abs(ov_fps - fps) > 0.05:
            print(f"FEHLER: {os.path.basename(ov_path)} hat {ov_fps:g} fps, Basis "
                  f"{fps:g} fps — Ergebnis waere zeitlich falsch. Quellen angleichen.",
                  file=sys.stderr)
            return 2
        if ov.shape[1:3] != (h, w):
            print(f"FEHLER: {ov_path} hat {ov.shape[2]}x{ov.shape[1]}, "
                  f"Basis {w}x{h}.", file=sys.stderr)
            return 2
        k = min(n, len(ov))
        if crop_mask is not None:
            m = crop_mask[:k]
        else:
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
