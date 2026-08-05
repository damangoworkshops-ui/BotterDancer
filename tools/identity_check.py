#!/usr/bin/env python
"""BotterDancer — Identitaets-Drift messen (CLIP-Vision statt Farb-Histogramm).

Ersetzt das am 04.08. ehrlich beerdigte Farb-Gate: Bhattacharyya-Distanzen auf
HS-Histogrammen konnten FORM-Drift nicht sehen (das korrekte cam180 mass 0.50
zur eigenen Referenz, die falsche Katzen-Variante nur 0.36 — die Metrik hat
also nicht diskriminiert). CLIP-Vision-Embeddings erfassen dagegen, WAS auf
dem Bild ist, nicht nur welche Farben.

Gemessen wird die Figur, nicht das Bild: der Ausschnitt kommt aus der
Pose-Bounding-Box (dieselbe Logik wie crew_composite --overlay-pose), sonst
dominiert der Hintergrund die Aehnlichkeit.

  python identity_check.py --video clip.mp4 --ref C:\\ComfyUI\\input\\ref.png
                           [--pose-dir <track>] [--min-sim 0.75]

Ausgabe: Kosinus-Aehnlichkeit je Stichprobenframe gegen die Referenz plus
Verlauf ueber den Clip (Drift von Anfang zu Ende). Exit 4, wenn die mittlere
Aehnlichkeit unter --min-sim liegt.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\ComfyUI")


def load_clip_vision():
    import comfy.clip_vision as cv
    import folder_paths
    names = folder_paths.get_filename_list("clip_vision")
    if not names:
        print("FEHLER: kein clip_vision-Modell in ComfyUI.", file=sys.stderr)
        sys.exit(2)
    path = folder_paths.get_full_path("clip_vision", names[0])
    return cv.load(path), names[0]


def embed(model, img_bgr):
    """(H,W,3) BGR uint8 -> (D,) normiertes Embedding."""
    import torch
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(rgb))[None]  # (1,H,W,3), wie ComfyUI IMAGE
    out = model.encode_image(t)
    v = out.image_embeds if hasattr(out, "image_embeds") else out["image_embeds"]
    v = v.detach().float().cpu().numpy().reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def pose_box(pose_png, shape, margin=0.12):
    import cv2
    img = cv2.imread(pose_png)
    if img is None:
        return None
    h, w = shape
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(img.max(axis=2) > 20)
    if len(xs) < 10:
        return None
    mx, my = int(margin * w), int(margin * h)
    return (max(0, xs.min() - mx), min(w, xs.max() + mx),
            max(0, ys.min() - my), min(h, ys.max() + my))


def main():
    ap = argparse.ArgumentParser(description="Identitaets-Drift per CLIP-Vision")
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref", required=True, help="Referenzbild der Figur")
    ap.add_argument("--pose-dir", help="Track-Pose-Verzeichnis — schneidet die Figur "
                                       "aus, statt das ganze Bild zu vergleichen")
    ap.add_argument("--samples", type=int, default=9)
    ap.add_argument("--min-sim", type=float, default=0.75,
                    help="Mindest-Kosinus-Aehnlichkeit zur Referenz. Kalibriert an echten "
                         "Laeufen (06.08.): korrekte Figuren 0.78-0.87, echte "
                         "Fehlbesetzung 0.41. WICHTIG: --pose-dir NUR bei Vollbild-"
                         "Renders angeben; bei figurzentrierten Crops schneidet die Box "
                         "ein zweites Mal und drueckt die Werte auf 0.72-0.75.")
    args = ap.parse_args()

    import cv2
    model, name = load_clip_vision()
    ref_img = cv2.imread(args.ref)
    if ref_img is None:
        print(f"FEHLER: Referenz {args.ref} nicht lesbar.", file=sys.stderr)
        return 2
    ref_vec = embed(model, ref_img)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    poses = sorted(glob.glob(os.path.join(args.pose_dir, "*.png"))) if args.pose_dir else []
    idxs = np.linspace(0, total - 1, args.samples).astype(int)
    sims, used = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame
        if poses and i < len(poses):
            box = pose_box(poses[int(i)], frame.shape[:2])
            if box:
                x0, x1, y0, y1 = box
                if (x1 - x0) > 32 and (y1 - y0) > 32:
                    crop = frame[y0:y1, x0:x1]
        sims.append(float(np.dot(ref_vec, embed(model, crop))))
        used.append(int(i))
    cap.release()
    if not sims:
        print("FEHLER: keine Frames lesbar.", file=sys.stderr)
        return 2

    sims = np.array(sims)
    print(f"[ident] {name} vs {os.path.basename(args.ref)} "
          f"({'Figur-Ausschnitt' if poses else 'ganzes Bild'})")
    for i, s in zip(used, sims):
        bar = "#" * int(max(0.0, s) * 40)
        print(f"  Frame {i:4d}: {s:.3f} {bar}")
    half = len(sims) // 2 or 1
    drift = float(sims[half:].mean() - sims[:half].mean())
    print(f"[ident] Mittel {sims.mean():.3f}, min {sims.min():.3f}, "
          f"Verlauf Anfang->Ende {drift:+.3f}")
    if sims.mean() < args.min_sim:
        print(f"ERGEBNIS: Identitaet zu schwach (Mittel {sims.mean():.3f} < "
              f"{args.min_sim}) — falsche Referenz oder Drift.", file=sys.stderr)
        return 4
    print("ERGEBNIS: Identitaet haelt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
