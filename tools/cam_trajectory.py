"""BotterDancer — Kamera-Trajektorien fuer One-Shots (virtuelle Steadicam/Crane).

Eigenes Modul (statt inline in reproject_camera.py), damit die Mathematik ohne
GVHMR-Stack testbar ist. Eine Trajektorie ist ein geeaster Verlauf eines
Kamera-Parameters (Azimut/Elevation/Distanz-Skala) ueber die VOLLE
Quell-Sequenz — das Resampling waehlt danach dieselben Frames wie beim
Pose-Rendern, Kamera und Motion bleiben also exakt synchron.
"""
import numpy as np

EASINGS = ("smoothstep", "linear")


def smoothstep(u):
    """Klassisches 3u^2-2u^3: Beschleunigen/Abbremsen wie eine gefuehrte Kamera,
    Ableitung 0 an beiden Enden (kein Ruck am Shot-Anfang/-Ende)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def trajectory(n_frames, start, end=None, easing="smoothstep"):
    """(n_frames,)-Verlauf von start nach end. end=None -> statisch (Altverhalten)."""
    if easing not in EASINGS:
        raise ValueError(f"easing {easing!r} nicht in {EASINGS}")
    if n_frames < 1:
        raise ValueError("n_frames muss >= 1 sein")
    if end is None or end == start or n_frames == 1:
        return np.full(n_frames, float(start))
    u = np.linspace(0.0, 1.0, n_frames)
    e = smoothstep(u) if easing == "smoothstep" else u
    return float(start) + (float(end) - float(start)) * e
