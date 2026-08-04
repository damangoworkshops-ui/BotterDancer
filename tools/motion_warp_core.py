"""BotterDancer — Kern-Mathematik des Motion-Time-Warp (numpy-only, testbar).

Choreographer-Stufe 1 (Produktentscheidung 04.07.: Moves auf das Beat-Grid des
NEUEN Songs syncen, Time-Warp auf Ziel-BPM, +/-15%% als natuerliche Grenze;
Review-Fix #7: Reihenfolge Rotationen -> Time-Warp -> Kontakte -> IK).

Bausteine:
- find_accents():  Bewegungs-Akzente = lokale Minima der Gelenk-Gesamtgeschwindigkeit
                   (ein "Hit" ist ein kurzer Stopp), mit Mindestabstand.
- estimate_period(): robustes Eigentempo der Choreo aus den Akzent-Intervallen.
- build_warp_map(): monotone stueckweise-lineare Zeit-Abbildung dst->src, die
                   Akzente auf die naechsten Ziel-Beats zieht; lokale Steigung
                   hart auf [1-max_stretch, 1+max_stretch] begrenzt (Anker, die
                   das verletzen wuerden, werden verworfen und gezaehlt).
- warp_time():     Abbildung anwenden (np.interp ueber die Anker).
- slerp():         batched Quaternion-Slerp fuer die Rotations-Resamples.
"""
import numpy as np


def moving_average(x, w=5):
    if w <= 1:
        return np.asarray(x, dtype=float)
    pad = w // 2
    xp = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def find_accents(speed, fps, min_gap_s=0.25, percentile=60.0):
    """Akzent-Zeiten (Sekunden) aus einem (F,)-Geschwindigkeitsverlauf.
    Lokales Minimum unterhalb des Perzentils; bei Kollision im Mindestabstand
    gewinnt das tiefere Minimum."""
    s = moving_average(speed, 5)
    thresh = np.percentile(s, percentile)
    cand = [i for i in range(1, len(s) - 1)
            if s[i] < thresh and s[i] <= s[i - 1] and s[i] <= s[i + 1]]
    kept = []
    for i in cand:
        if kept and (i - kept[-1]) / fps < min_gap_s:
            if s[i] < s[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    return np.array(kept, dtype=float) / fps


def estimate_period(accents_s):
    """Median-Akzent-Intervall in Sekunden; None bei <3 Akzenten.
    ACHTUNG (empirisch 04.08.): fuer Tempo-Schaetzung UNGEEIGNET, wenn die
    Choreo Raster-Slots ueberspringt (Median mischt 1x/2x/3x-Vielfache) —
    dafuer estimate_tatum() nutzen."""
    if len(accents_s) < 3:
        return None
    return float(np.median(np.diff(accents_s)))


def estimate_tatum(accents_s, lo=0.25, hi=0.8, step=0.002):
    """Feinstes gemeinsames Raster (Tatum) der Akzente: Scan ueber Periode+Phase,
    Score = mittlerer Restfehler relativ zur Periode (sonst gewinnt trivial die
    kleinste Periode). Returns (period_s, offset_s, resid_s)."""
    accents = np.asarray(accents_s, dtype=float)
    best = None
    for p in np.arange(lo, hi + 1e-9, step):
        off, resid = best_grid_offset(accents, float(p), steps=96)
        score = resid / p
        if best is None or score < best[0]:
            best = (score, float(p), off, resid)
    _, p, off, resid = best
    return p, off, resid


def select_strongest(accents_s, speed, fps, k):
    """Die k staerksten Akzente (tiefste Taeler relativ zur lokalen Umgebung).
    Tanz-Realitaet: die STARKEN Hits landen auf dem Beat, nicht jedes lokale
    Minimum — schwache Neben-Minima verwaessern sowohl Matching als auch Metrik."""
    if len(accents_s) <= k:
        return np.asarray(accents_s, dtype=float)
    s = moving_average(speed, 5)
    depth = []
    for a in accents_s:
        i = int(round(a * fps))
        lo, hi = max(0, i - int(0.3 * fps)), min(len(s), i + int(0.3 * fps) + 1)
        depth.append(float(np.median(s[lo:hi]) - s[min(i, len(s) - 1)]))
    idx = np.argsort(depth)[::-1][:k]
    return np.sort(np.asarray(accents_s, dtype=float)[idx])


def best_grid_offset(accents_s, period, steps=240):
    """Phasenlage des Beat-Grids, die die Akzente am besten trifft.
    Wir ERZEUGEN den Track — also gehoert das Grid zu den Akzenten gedreht,
    nicht umgekehrt (ohne das verwirft die Steigungsschranke frueh kaskadierend).
    Returns (offset_s, mittlerer Restfehler_s)."""
    accents = np.asarray(accents_s, dtype=float)
    cands = np.linspace(0.0, period, steps, endpoint=False)
    errs = []
    for o in cands:
        r = np.mod(accents - o, period)
        errs.append(float(np.minimum(r, period - r).mean()))
    i = int(np.argmin(errs))
    return float(cands[i]), errs[i]


def build_warp_map(accents_s, duration_s, target_beats, max_stretch=0.15):
    """Anker (dst, src), monoton in beiden Achsen, Steigungsschranke erzwungen.

    Matching: jeder Akzent zieht auf den naechstgelegenen NOCH FREIEN, spaeteren
    Ziel-Beat (monoton). Ein Anker wird verworfen, wenn das Segment zu ihm eine
    Steigung ausserhalb [1-max_stretch, 1+max_stretch] braeuchte.
    Returns: anchors (N,2) mit Spalten [dst, src], n_matched, n_dropped.
    """
    lo, hi = 1.0 - max_stretch, 1.0 + max_stretch
    anchors = [(0.0, 0.0)]
    dropped = 0
    last_beat_idx = -1
    beats = np.asarray(target_beats, dtype=float)
    for a in np.asarray(accents_s, dtype=float):
        if a <= anchors[-1][1]:
            continue
        usable = np.where(beats > anchors[-1][0])[0]
        usable = usable[usable > last_beat_idx]
        if len(usable) == 0:
            break
        j = usable[np.argmin(np.abs(beats[usable] - a))]
        dst = float(beats[j])
        d_dst, d_src = dst - anchors[-1][0], a - anchors[-1][1]
        if d_dst <= 0 or not (lo <= d_src / d_dst <= hi):
            dropped += 1
            continue
        anchors.append((dst, float(a)))
        last_beat_idx = j
    # Endanker: Rest der Sequenz mit Steigung 1 auslaufen lassen
    tail = duration_s - anchors[-1][1]
    if tail > 1e-6:
        anchors.append((anchors[-1][0] + tail, duration_s))
    arr = np.array(anchors, dtype=float)
    assert np.all(np.diff(arr[:, 0]) > 0) and np.all(np.diff(arr[:, 1]) > 0), \
        "Warp-Map muss strikt monoton sein"
    return arr, len(anchors) - 2 + (0 if tail > 1e-6 else 1), dropped


def warp_time(anchors, t_dst):
    """Ziel-Zeit(en) -> Quell-Zeit(en) ueber die Anker interpolieren."""
    return np.interp(np.asarray(t_dst, dtype=float), anchors[:, 0], anchors[:, 1])


def axis_angle_to_quat(aa):
    """(..., 3) Achse*Winkel -> (..., 4) Einheitsquaternion (w, x, y, z)."""
    aa = np.asarray(aa, dtype=float)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.where(small, 0.0, aa / np.where(small, 1.0, angle))
    half = angle / 2.0
    q = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
    return np.where(small, np.array([1.0, 0.0, 0.0, 0.0]), q)


def quat_to_axis_angle(q):
    """(..., 4) Einheitsquaternion -> (..., 3) Achse*Winkel, Winkel in [0, pi]."""
    q = np.asarray(q, dtype=float)
    q = np.where(q[..., :1] < 0, -q, q)  # w >= 0 -> kuerzeste Darstellung
    w = q[..., :1].clip(-1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt((1.0 - w * w).clip(min=0.0))
    small = s < 1e-8
    axis = np.where(small, 0.0, q[..., 1:] / np.where(small, 1.0, s))
    return axis * angle


def slerp(q0, q1, u):
    """Batched Quaternion-Slerp. q0/q1: (..., 4) Einheitsquaternionen, u: Skalar
    oder (...,). Vorzeichen-korrigiert (kuerzester Bogen), stabil bei kleinen
    Winkeln (lerp-Fallback)."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    u = np.asarray(u, dtype=float)
    while u.ndim < q0.ndim:  # (F,) gegen (F,J,4) broadcastbar machen
        u = u[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    w0 = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(small, 1.0, sin_t))
    w1 = np.where(small, u, np.sin(u * theta) / np.where(small, 1.0, sin_t))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-12)
