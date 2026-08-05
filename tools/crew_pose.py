#!/usr/bin/env python
"""BotterDancer — Full-Crew-Casting: Multi-Person-Pose fuer Gruppenchoreographie.

Gegenstueck zu filter_pose_v2 (das EINE Person isoliert). Hier bleiben mehrere
Taenzerinnen erhalten, werden aber stabil ueber die Sequenz getrackt — sonst
wechseln Nachbar-Skelette die Identitaet und Wan-Animate rendert Figuren, die
ineinander morphen.

- Auswahl: die N groessten Personen pro Frame (Hintergrund-Zuschauer und
  Spiegel-Reflexionen fliegen raus).
- Tracking: greedy Nearest-Neighbour auf dem Torso-Anker (Neck+Hips), mit
  Max-Jump-Gate; Luecken werden pro Track interpoliert.
- Ausgabe: eine PNG-Sequenz mit ALLEN Tracks (Gruppen-Render) und/oder mit
  --split je Track eine eigene Sequenz (fuer per-Person-Referenzen).

  python crew_pose.py --src pose.json --outdir <dir> --crew 3 [--split]

Der 2D-Pfad uebernimmt die ORIGINAL-KAMERA automatisch: die Skelette bewegen
sich mit ihr. Fuer Gruppen ist das der richtige Pfad (GVHMR trackt nur eine
Person, und eine 3D-Reprojektion wuerde die Kamerafuehrung des Originals
gerade wegwerfen).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, r"C:\ComfyUI\custom_nodes\comfyui_controlnet_aux\src")
from custom_controlnet_aux.dwpose import decode_json_as_poses, draw_poses  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

NECK, R_HIP, L_HIP = 1, 8, 11


def kps(person):
    return np.array(person["pose_keypoints_2d"], dtype=float).reshape(-1, 3)


def anchor(person):
    k = kps(person)
    pts = k[[NECK, R_HIP, L_HIP]]
    if np.any(pts[:, 2] <= 0):
        valid = k[k[:, 2] > 0]
        return valid[:, :2].mean(axis=0) if len(valid) >= 3 else None
    return pts[:, :2].mean(axis=0)


def bbox_area(person):
    k = kps(person)
    v = k[k[:, 2] > 0]
    if len(v) < 3:
        return 0.0
    return float((v[:, 0].max() - v[:, 0].min()) * (v[:, 1].max() - v[:, 1].min()))


def lerp_person(a, b, t):
    out = {}
    for key in a:
        va, vb = a.get(key), b.get(key)
        if key.endswith("_keypoints_2d") and isinstance(va, list) and isinstance(vb, list) \
                and len(va) == len(vb):
            x, y = np.array(va, dtype=float), np.array(vb, dtype=float)
            v = x * (1 - t) + y * t
            v[2::3] = np.minimum(x[2::3], y[2::3])
            out[key] = v.tolist()
        else:
            out[key] = va
    return out


def build_tracks(frames, crew, max_jump_frac=0.18):
    """[track][frame] -> person|None. Greedy-Zuordnung auf Torso-Ankern."""
    W, H = frames[0]["canvas_width"], frames[0]["canvas_height"]
    diag = float(np.hypot(W, H))
    tracks = [[None] * len(frames) for _ in range(crew)]
    last = [None] * crew  # letzter bekannter Anker je Track
    for i, f in enumerate(frames):
        people = sorted(f.get("people", []), key=bbox_area, reverse=True)[:crew]
        cands = [(p, anchor(p)) for p in people]
        cands = [(p, a) for p, a in cands if a is not None]
        if i == 0 or all(l is None for l in last):
            # Seed: nach x-Position sortieren -> stabile Links-nach-rechts-Reihenfolge
            for t, (p, a) in enumerate(sorted(cands, key=lambda c: c[1][0])[:crew]):
                tracks[t][i] = p
                last[t] = a
            continue
        free = list(range(len(cands)))
        pairs = []
        for t in range(crew):
            if last[t] is None:
                continue
            for ci in free:
                pairs.append((float(np.linalg.norm(cands[ci][1] - last[t])), t, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for d, t, ci in pairs:
            if t in used_t or ci in used_c or d > max_jump_frac * diag:
                continue
            tracks[t][i] = cands[ci][0]
            last[t] = cands[ci][1]
            used_t.add(t)
            used_c.add(ci)
        # uebrige Kandidaten an noch leere Tracks (Wiedereinstieg nach Verdeckung)
        for t in range(crew):
            if t in used_t:
                continue
            rest = [ci for ci in range(len(cands)) if ci not in used_c]
            if rest:
                ci = rest[0]
                tracks[t][i] = cands[ci][0]
                last[t] = cands[ci][1]
                used_c.add(ci)
    return tracks


def fill_gaps(track, max_gap):
    """Luecken <= max_gap interpolieren, Randluecken halten. (n_interp, n_hold)"""
    n_i = n_h = 0
    i = 0
    while i < len(track):
        if track[i] is None:
            j = i
            while j < len(track) and track[j] is None:
                j += 1
            prev_ok, next_ok = i > 0 and track[i - 1] is not None, j < len(track)
            if (j - i) <= max_gap and prev_ok and next_ok:
                for k in range(i, j):
                    track[k] = lerp_person(track[i - 1], track[j], (k - i + 1) / (j - i + 1))
                    n_i += 1
            else:
                hold = track[j] if next_ok else (track[i - 1] if prev_ok else None)
                for k in range(i, j):
                    track[k] = hold
                    n_h += 1
            i = j
        else:
            i += 1
    return n_i, n_h


def render(frames_people, W, H, outdir):
    tmp = outdir.rstrip("\\/") + f".tmp-{os.getpid()}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for i, people in enumerate(frames_people):
        ff = {"people": [p for p in people if p], "canvas_width": W, "canvas_height": H}
        poses, _, h, w = decode_json_as_poses(ff)
        img = draw_poses(poses, h, w, True, True, True)
        Image.fromarray(img).save(os.path.join(tmp, f"pose_{i:04d}.png"))
    n = len([e for e in os.listdir(tmp) if e.endswith(".png")])
    if n != len(frames_people):
        print(f"FEHLER: {n}/{len(frames_people)} PNGs gerendert.", file=sys.stderr)
        sys.exit(3)
    if os.path.isdir(outdir):
        foreign = [e for e in os.listdir(outdir) if not e.startswith("pose_")]
        if foreign:
            print(f"FEHLER: {outdir} enthaelt fremde Eintraege {foreign[:3]}.", file=sys.stderr)
            sys.exit(3)
        shutil.rmtree(outdir)
    os.rename(tmp, outdir)
    return n


def main():
    ap = argparse.ArgumentParser(description="Multi-Person-Pose fuer Gruppenchoreo")
    ap.add_argument("--src", required=True, help="Pose-JSON (SavePoseKpsAsJsonFile)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--crew", type=int, required=True, help="Anzahl Taenzer:innen")
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--split", action="store_true",
                    help="zusaetzlich je Track eine eigene Sequenz (<outdir>_t0, _t1, ...)")
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8-sig") as f:
        frames = json.load(f)
    W, H = frames[0]["canvas_width"], frames[0]["canvas_height"]
    print(f"[crew] {len(frames)} Frames, Canvas {W}x{H}, Ziel-Crew {args.crew}")

    tracks = build_tracks(frames, args.crew)
    for t, tr in enumerate(tracks):
        found = sum(1 for p in tr if p is not None)
        n_i, n_h = fill_gaps(tr, args.max_gap)
        print(f"  Track {t}: {found}/{len(frames)} erkannt, {n_i} interpoliert, {n_h} gehalten")
        if found < 0.5 * len(frames):
            print(f"WARNUNG: Track {t} ueberwiegend synthetisch — --crew zu hoch?",
                  file=sys.stderr)

    n = render([[tracks[t][i] for t in range(args.crew)] for i in range(len(frames))],
               W, H, args.outdir)
    print(f"DONE: {n} Frames (Crew) -> {args.outdir}")

    if args.split:
        for t in range(args.crew):
            d = args.outdir.rstrip("\\/") + f"_t{t}"
            render([[tracks[t][i]] for i in range(len(frames))], W, H, d)
            print(f"DONE: Track {t} -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
