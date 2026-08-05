"""BotterDancer — Bewegungs-Akzente aus 2D-Posen (numpy-only, testbar).

motion_warp.find_accents arbeitet auf 3D-Gelenken aus GVHMR — das gibt es nur
im Solo-Pfad. Fuer die Segmentwahl bei echten Songs (und fuer Gruppen) braucht
es dieselbe Information aus den 2D-Keypoints, die DWPose ohnehin liefert.

Ein Akzent ist ein kurzer Stopp: lokales Minimum der mittleren
Keypoint-Geschwindigkeit. Nur konfidente Punkte zaehlen, und die Geschwindigkeit
wird auf die Koerpergroesse normiert — sonst haengt die Schwelle an der
Bildaufloesung und daran, wie weit die Kamera weg steht.
"""
import json

import numpy as np

NECK, R_HIP, L_HIP = 1, 8, 11


def _kps(person):
    return np.array(person["pose_keypoints_2d"], dtype=float).reshape(-1, 3)


def body_scale(person):
    """Torso-Laenge als Groessenmass (robuster als die Bounding-Box, die bei
    ausgestreckten Armen springt)."""
    k = _kps(person)
    if k[NECK, 2] <= 0:
        return None
    hips = [i for i in (R_HIP, L_HIP) if k[i, 2] > 0]
    if not hips:
        return None
    mid = k[hips, :2].mean(axis=0)
    d = float(np.linalg.norm(k[NECK, :2] - mid))
    return d if d > 1e-3 else None


def speed_series(frames, person_index=0):
    """(F,) mittlere Keypoint-Geschwindigkeit in Koerperlaengen pro Frame."""
    seq = []
    for fr in frames:
        people = fr.get("people") or []
        if person_index < len(people):
            seq.append(people[person_index])
        else:
            seq.append(None)
    speeds = [0.0]
    for a, b in zip(seq, seq[1:]):
        if a is None or b is None:
            speeds.append(speeds[-1])
            continue
        ka, kb = _kps(a), _kps(b)
        ok = (ka[:, 2] > 0) & (kb[:, 2] > 0)
        s = body_scale(b) or body_scale(a)
        if not ok.any() or not s:
            speeds.append(speeds[-1])
            continue
        d = np.linalg.norm(kb[ok, :2] - ka[ok, :2], axis=-1).mean()
        speeds.append(float(d / s))
    speeds[0] = speeds[1] if len(speeds) > 1 else 0.0
    return np.array(speeds)


def accents_from_frames(frames, fps, min_gap_s=0.25, percentile=60.0, person_index=0):
    """Akzent-Zeiten (Sekunden) aus einer Pose-Sequenz."""
    from motion_warp_core import find_accents
    return find_accents(speed_series(frames, person_index), fps,
                        min_gap_s=min_gap_s, percentile=percentile)


def accents_from_json(path, fps, **kw):
    with open(path, "r", encoding="utf-8-sig") as f:
        frames = json.load(f)
    return accents_from_frames(frames, fps, **kw)
