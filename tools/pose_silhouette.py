#!/usr/bin/env python
"""BotterDancer — Personen-Silhouetten aus Pose-Keypoints (fuer Room-Keeper).

Wan-Animates character_mask entscheidet pro Pixel: 1 = hier NEU generieren,
0 = aus dem background_video uebernehmen (am Node-Code verifiziert 06.08.,
nodes_wan.py: mask_refmotion startet als ones und wird von character_mask
ueberschrieben). Fuer den Room-Keeper heisst das: die Maske muss die
ORIGINAL-Taenzerin grosszuegig abdecken, damit sie durch die neue Figur
ersetzt wird — der unmaskierte Rest bleibt der echte Raum.

Eine duenne Skelett-Linie (ImageToMask auf dem Pose-PNG) genuegt dafuer NICHT.
Dieses Tool malt eine gefuellte Silhouette: Torso-Polygon, Gliedmassen als
dicke Balken, Kopf als Kreis — danach dilatiert, damit auch voluminoese
Figuren (Fell, Kostuem, Fluegel) Platz haben.

  python pose_silhouette.py --src pose.json --outdir <dir> [--track 0]
                            [--dilate 28] [--limb 26] [--head 34]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

# OpenPose-18
NOSE, NECK = 0, 1
R_SHO, R_ELB, R_WRI = 2, 3, 4
L_SHO, L_ELB, L_WRI = 5, 6, 7
R_HIP, R_KNE, R_ANK = 8, 9, 10
L_HIP, L_KNE, L_ANK = 11, 12, 13
LIMBS = [(NECK, R_SHO), (R_SHO, R_ELB), (R_ELB, R_WRI),
         (NECK, L_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI),
         (NECK, R_HIP), (R_HIP, R_KNE), (R_KNE, R_ANK),
         (NECK, L_HIP), (L_HIP, L_KNE), (L_KNE, L_ANK)]
TORSO = [R_SHO, L_SHO, L_HIP, R_HIP]


def kps(person):
    return np.array(person["pose_keypoints_2d"], dtype=float).reshape(-1, 3)


def silhouette(people, w, h, limb, head, dilate):
    import cv2
    m = np.zeros((h, w), np.uint8)
    for p in people:
        if not p:
            continue
        k = kps(p)
        ok = k[:, 2] > 0
        for a, b in LIMBS:
            if ok[a] and ok[b]:
                cv2.line(m, tuple(np.int32(k[a, :2])), tuple(np.int32(k[b, :2])),
                         255, limb, cv2.LINE_AA)
        pts = [np.int32(k[i, :2]) for i in TORSO if ok[i]]
        if len(pts) >= 3:
            cv2.fillConvexPoly(m, cv2.convexHull(np.array(pts)), 255)
        if ok[NECK]:
            centre = k[NOSE, :2] if ok[NOSE] else k[NECK, :2]
            cv2.circle(m, tuple(np.int32(centre)), head, 255, -1, cv2.LINE_AA)
        # Fuesse etwas verlaengern: Schuhe/Pfoten liegen unter dem Knoechel
        for ank, kne in ((R_ANK, R_KNE), (L_ANK, L_KNE)):
            if ok[ank] and ok[kne]:
                d = k[ank, :2] - k[kne, :2]
                n = np.linalg.norm(d)
                if n > 1:
                    tip = k[ank, :2] + d / n * (limb * 0.8)
                    cv2.line(m, tuple(np.int32(k[ank, :2])), tuple(np.int32(tip)),
                             255, limb, cv2.LINE_AA)
    if dilate > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
        m = cv2.dilate(m, ker)
        m = cv2.GaussianBlur(m, (2 * (dilate // 3) + 1,) * 2, 0)
    return m


def main():
    ap = argparse.ArgumentParser(description="Silhouetten-Masken aus Pose-Keypoints")
    ap.add_argument("--src", required=True, help="Pose-JSON (SavePoseKpsAsJsonFile)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--track", type=int, help="nur diese Person (Index nach Groesse); "
                                              "ohne Angabe: ALLE Personen im Frame")
    ap.add_argument("--limb", type=int, default=26, help="Gliedmassen-Dicke (px)")
    ap.add_argument("--head", type=int, default=34, help="Kopf-Radius (px)")
    ap.add_argument("--dilate", type=int, default=28, help="Dilatation (px) — Puffer fuer "
                                                          "voluminoese Figuren")
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("FEHLER: OpenCV noetig.", file=sys.stderr)
        return 2
    import cv2
    from PIL import Image

    with open(args.src, "r", encoding="utf-8-sig") as f:
        frames = json.load(f)
    w, h = frames[0]["canvas_width"], frames[0]["canvas_height"]

    def area(p):
        k = kps(p)
        v = k[k[:, 2] > 0]
        return 0.0 if len(v) < 3 else float((v[:, 0].max() - v[:, 0].min()) *
                                            (v[:, 1].max() - v[:, 1].min()))

    tmp = args.outdir.rstrip("\\/") + f".tmp-{os.getpid()}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    covers = []
    for i, fr in enumerate(frames):
        people = fr.get("people", [])
        if args.track is not None:
            people = sorted(people, key=area, reverse=True)
            people = [people[args.track]] if args.track < len(people) else []
        m = silhouette(people, w, h, args.limb, args.head, args.dilate)
        covers.append(float((m > 127).mean()))
        Image.fromarray(m).convert("L").save(os.path.join(tmp, f"mask_{i:04d}.png"))

    if os.path.isdir(args.outdir):
        foreign = [e for e in os.listdir(args.outdir) if not e.startswith("mask_")]
        if foreign:
            print(f"FEHLER: {args.outdir} enthaelt fremde Eintraege {foreign[:3]}.",
                  file=sys.stderr)
            return 3
        shutil.rmtree(args.outdir)
    os.rename(tmp, args.outdir)
    print(f"DONE: {len(frames)} Masken -> {args.outdir} "
          f"(Abdeckung im Mittel {np.mean(covers) * 100:.1f}%, "
          f"max {np.max(covers) * 100:.1f}%)")
    if np.mean(covers) < 0.01:
        print("WARNUNG: Maske sehr klein — --dilate/--limb erhoehen?", file=sys.stderr)
    if np.mean(covers) > 0.6:
        print("WARNUNG: Maske deckt fast alles ab — der Raum wuerde neu erfunden.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
