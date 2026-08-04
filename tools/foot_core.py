"""BotterDancer — Kontakt-Rekonstruktion + Root-Anchoring (numpy-only, testbar).

Choreographer-Stufe 2 (Review-Fix #7: Rotationen -> Time-Warp -> KONTAKTE NEU ->
Foot-IK). Dieses Modul macht die Kontakt-Erkennung und die erste Fix-Stufe:
ROOT-Anchoring — die horizontale Wurzel-Trajektorie wird so korrigiert, dass der
Standfuss waehrend eines Kontakts planted bleibt. Das entfernt den Skate-Anteil,
der aus GVHMR-Trajektorien-Drift stammt, OHNE Gelenkwinkel anzufassen (kein
Naturalness-Risiko). Echtes Zwei-Knochen-Foot-IK (Restfehler in den Beinen)
ist die naechste Stufe, falls die Skate-Metrik es noch verlangt.

Konventionen: Y-up-Welt (wie GVHMR smpl_params_global), xz = horizontal.
"""
import numpy as np


def moving_average_2d(x, w):
    """(F, 2) spaltenweise Box-Filter mit Randfortsetzung."""
    if w <= 1:
        return np.asarray(x, dtype=float)
    pad = w // 2
    xp = np.pad(np.asarray(x, dtype=float), ((pad, pad), (0, 0)), mode="edge")
    kern = np.ones(w) / w
    return np.stack([np.convolve(xp[:, c], kern, mode="valid")
                     for c in range(x.shape[1])], axis=1)


def detect_contacts(foot_y, foot_vxz, floor_y, h_thresh=0.06, v_thresh=0.35):
    """Bodenkontakt pro Frame: niedrig UND langsam.
    foot_y: (F,) Hoehe; foot_vxz: (F,) horizontale Geschwindigkeit (m/s).
    Die Geschwindigkeitsbedingung verhindert, dass ein tiefer Swing-Durchgang
    als Kontakt zaehlt (dann wuerde das Anchoring die Schwungphase anhalten)."""
    return (np.asarray(foot_y) - floor_y < h_thresh) & (np.asarray(foot_vxz) < v_thresh)


def contact_segments(mask, min_len=3):
    """[(start, end_exklusiv)] zusammenhaengende Kontaktabschnitte; kuerzere als
    min_len Frames werden verworfen (Detektions-Flackern)."""
    mask = np.asarray(mask, dtype=bool)
    segs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segs.append((start, len(mask)))
    return segs


def clean_contacts(mask, min_len=3, close_gaps=2):
    """Maske bereinigen: erst kurze LUECKEN schliessen (Detektions-Dropouts —
    sonst resettet ein 1-Frame-Loch den Anker und re-plantet den Fuss an der
    gedrifteten Position, Verifier-Befund 05.08.), dann kurze Segmente verwerfen."""
    mask = np.asarray(mask, dtype=bool).copy()
    if close_gaps > 0:
        inv = ~mask
        for a, b in contact_segments(inv, min_len=1):
            if 0 < a and b < len(mask) and (b - a) <= close_gaps:
                mask[a:b] = True
    out = np.zeros(len(mask), dtype=bool)
    for a, b in contact_segments(mask, min_len):
        out[a:b] = True
    return out


def root_anchor_correction(feet_xz, contacts, fps, smooth_s=0.12):
    """Horizontale Root-Korrektur (F, 2), sodass Standfuesse planted bleiben.

    Pro Kontaktbeginn wird der Fuss an seiner (bereits korrigierten) Position
    verankert; waehrend des Kontakts fordert er die Korrektur, die ihn dort
    haelt. Bei Doppelkontakt werden die Forderungen gemittelt, ohne Kontakt
    wird die letzte Korrektur gehalten (kein Zurueckschnappen). Am Ende
    geglaettet (Box-Filter), damit Kontaktwechsel keine Spruenge erzeugen —
    das weicht die exakte Plantung minimal auf; die Skate-Metrik entscheidet.

    feet_xz: (F, n_feet, 2); contacts: (F, n_feet) bool (bereits bereinigt).
    """
    feet_xz = np.asarray(feet_xz, dtype=float)
    contacts = np.asarray(contacts, dtype=bool)
    n, n_feet = contacts.shape
    corr = np.zeros((n, 2))
    anchors = [None] * n_feet
    prev = np.zeros(2)
    for t in range(n):
        # Verifier-Befund 05.08. (HOCH): bei jeder Aenderung der Kontaktmenge
        # ALLE aktiven Fuesse an ihrer korrigierten Ist-Position neu ankern —
        # sonst entlaedt sich die Doppelkontakt-Disparitaet beim Uebergang
        # Doppel->Einzel als Ein-Frame-Sprung.
        set_changed = t == 0 or not np.array_equal(contacts[t], contacts[t - 1])
        demands, idxs = [], []
        for f in range(n_feet):
            if contacts[t, f]:
                if anchors[f] is None or set_changed:
                    anchors[f] = feet_xz[t, f] + prev  # planted an korrigierter Position
                demands.append(anchors[f] - feet_xz[t, f])
                idxs.append(f)
            else:
                anchors[f] = None
        if demands:
            mean_d = np.mean(demands, axis=0)
            # Infeasiblen Anteil NICHT akkumulieren: Anker zur tatsaechlich
            # applizierten Korrektur nachfuehren. Der Rest (Fuesse divergieren
            # real) ist per Root-Translation prinzipiell unloesbar -> Stufe 3 IK.
            for f, d in zip(idxs, demands):
                anchors[f] = anchors[f] + (mean_d - d)
            prev = mean_d
        corr[t] = prev
    return moving_average_2d(corr, max(1, int(round(smooth_s * fps)) | 1))


def skate_metric(feet_xz, contacts, fps):
    """Mittlere horizontale Fussgeschwindigkeit (m/s) ueber Intervalle, deren
    BEIDE Endpunkte im Kontakt liegen (Verifier-Befund 05.08.: contacts[:-1]
    allein rechnete das Toe-off-Intervall ein und liess Heel-Strike-Slides aus
    — perfekte Plantung ergab nie 0). Gleicher Geschwindigkeits-Kern wie
    motion_warp.foot_skate, aber ANDERE Kontakt-Maske (dort hoehen-only ohne
    v-Gate) — Zahlen der beiden Tools nie direkt vergleichen."""
    feet_xz = np.asarray(feet_xz, dtype=float)
    contacts = np.asarray(contacts, dtype=bool)
    vals = []
    for f in range(feet_xz.shape[1]):
        v = np.linalg.norm(np.diff(feet_xz[:, f], axis=0), axis=-1) * fps
        m = contacts[:-1, f] & contacts[1:, f]
        if m.any():
            vals.append(float(v[m].mean()))
    return float(np.mean(vals)) if vals else float("nan")
