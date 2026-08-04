"""BotterDancer — Pose-Isolation v2.

Fixes gegenueber v1 (Deep-Review 2026-07-10):
- Torso-Anker-Tracker (EMA + Max-Jump-Gate) statt argmax(bbox_area) pro Frame
- Struktur-Gate: Neck + beide Hips Pflicht (Chimaeren-/Fragment-Filter)
- Gap-Handling: <= --max-gap Frames linear interpolieren, darueber lauter Fehler
- Bottom-Padding des Canvas (Fuesse bei y>canvas_height werden nicht mehr amputiert)
- OUTDIR wird geleert (kein Stale-Tail in der ffmpeg-Sequenz)
- Ehrliche Zaehler: fresh / interpolated / failed statt "49/49"
- --keep-all: rendert ALLE Personen (fuer den Single-Variable-Dilution-Test)
- meta.json-Sidecar: Selektion pro Frame, Modi, Parameter (Cache-/Repro-Grundlage)

Nutzung:
  python filter_pose_v2.py --src <pose.json> --outdir <dir> [--pad-bottom 48]
                           [--max-gap 8] [--max-jump 0.12] [--keep-all]
"""
import sys, json, os, argparse, shutil, datetime

sys.path.insert(0, r"C:\ComfyUI\custom_nodes\comfyui_controlnet_aux\src")
from custom_controlnet_aux.dwpose import decode_json_as_poses, draw_poses  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# OpenPose-18 Indizes
NECK, R_HIP, L_HIP = 1, 8, 11
ANCHOR_IDXS = (NECK, R_HIP, L_HIP)


def kps(person):
    return np.array(person["pose_keypoints_2d"], dtype=float).reshape(-1, 3)


def bbox_area(person):
    k = kps(person)
    valid = k[k[:, 2] > 0]
    if len(valid) < 3:
        return 0.0
    return float((valid[:, 0].max() - valid[:, 0].min()) * (valid[:, 1].max() - valid[:, 1].min()))


def anchor(person):
    """Torso-Anker = Mittel aus Neck/R-Hip/L-Hip. None, wenn Struktur-Gate reisst."""
    k = kps(person)
    pts = k[list(ANCHOR_IDXS)]
    if np.any(pts[:, 2] <= 0):
        return None
    return pts[:, :2].mean(axis=0)


def lerp_person(a, b, t):
    """Linear interpolierte Person (alle vorhandenen Keypoint-Arrays).

    None-sicher: DWPose schreibt fehlende Arrays (z.B. Haende bei
    include_hand=False) als null — len(None) wuerde crashen. None wird
    unveraendert kopiert; decode_json_as_poses faengt es ab."""
    out = {}
    for key in a:
        va_raw, vb_raw = a.get(key), b.get(key)
        if (key.endswith("_keypoints_2d") and isinstance(va_raw, list)
                and isinstance(vb_raw, list) and len(va_raw) == len(vb_raw)):
            va, vb = np.array(va_raw, dtype=float), np.array(vb_raw, dtype=float)
            v = va * (1 - t) + vb * t
            # Confidence konservativ: Minimum beider Seiten
            v[2::3] = np.minimum(va[2::3], vb[2::3])
            out[key] = v.tolist()
        else:
            out[key] = a[key]
    return out


def safe_remove_outdir(outdir):
    """OUTDIR entfernen — aber NUR, wenn er auch wirklich wie unser Output aussieht.
    Ein Tippfehler (--outdir C:\\ComfyUI\\input) darf nicht den ganzen Ordner
    vernichten (Audit P0-3). Seit 04.08. (F4) entfernt statt geleert: der neue
    fertige Satz wird per rename publiziert, nicht in-place geschrieben."""
    if os.path.lexists(outdir) and not os.path.isdir(outdir):
        print(f"FEHLER: --outdir ist eine DATEI, kein Ordner: {outdir}", file=sys.stderr)
        sys.exit(3)
    if not os.path.isdir(outdir):
        return
    entries = os.listdir(outdir)
    foreign = [e for e in entries
               if not (e.startswith("pose_") and e.lower().endswith(".png"))]
    if foreign:
        print(f"FEHLER: {outdir} enthaelt {len(foreign)} fremde Eintraege "
              f"(z.B. {foreign[:3]}) — Abbruch statt rmtree. "
              f"Anderes --outdir waehlen oder Ordner manuell leeren.", file=sys.stderr)
        sys.exit(3)
    shutil.rmtree(outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pad-bottom", type=int, default=48, help="Canvas unten erweitern (px, 16er-Vielfache halten)")
    ap.add_argument("--max-gap", type=int, default=8, help="max. Frames Luecke fuer Interpolation")
    ap.add_argument("--max-jump", type=float, default=0.12, help="Anker-Sprung-Gate als Anteil der Canvas-Diagonale")
    ap.add_argument("--lost-fallback", type=int, default=3, help="ab n Lost-Frames: Fallback largest-bbox + Reseed")
    ap.add_argument("--keep-all", action="store_true", help="alle Personen rendern (Dilution-Test)")
    ap.add_argument("--max-synth-share", type=float, default=0.3,
                    help="max. Anteil synthetischer Frames (interpolated/hold), sonst Abbruch")
    ap.add_argument("--fps", type=float, default=16.0,
                    help="fps der Quell-Keypoints (nur Metadaten/Repro; force_rate aus save_pose_81.json)")
    args = ap.parse_args()

    frames = json.load(open(args.src))
    W = frames[0]["canvas_width"]
    H = frames[0]["canvas_height"]
    H_out = H + args.pad_bottom
    diag = float(np.hypot(W, H))

    selection = []          # pro Frame: {frame, mode, person_idx, dist}
    chosen = [None] * len(frames)

    if args.keep_all:
        for i, f in enumerate(frames):
            chosen[i] = list(f.get("people", []))
            selection.append({"frame": i, "mode": "all", "n": len(chosen[i])})
    else:
        ema = None
        lost = 0
        v1_diff = 0
        for i, f in enumerate(frames):
            people = f.get("people", [])
            cands = [(idx, p, anchor(p)) for idx, p in enumerate(people)]
            cands = [(idx, p, a) for idx, p, a in cands if a is not None]
            pick, dist, mode = None, None, None

            if cands:
                if ema is None:
                    # Seed: Anker am naechsten zur Canvas-Mitte
                    center = np.array([W / 2.0, H / 2.0])
                    idx, p, a = min(cands, key=lambda c: np.linalg.norm(c[2] - center))
                    pick, dist, mode = (idx, p, a), float(np.linalg.norm(a - center)), "seed-center"
                else:
                    idx, p, a = min(cands, key=lambda c: np.linalg.norm(c[2] - ema))
                    d = float(np.linalg.norm(a - ema))
                    if d <= args.max_jump * diag:
                        pick, dist, mode = (idx, p, a), d, "track"

            if pick is None:
                lost += 1
                if lost >= args.lost_fallback and people:
                    p = max(people, key=bbox_area)
                    a = anchor(p)
                    if a is not None:
                        pick, dist, mode = (people.index(p), p, a), -1.0, "fallback-largest"
            if pick is not None:
                idx, p, a = pick
                chosen[i] = [p]
                # Nach Fallback VOLL reseeden: 50 % Alt-Anker liesse das
                # Max-Jump-Gate im Folgeframe sofort wieder reissen.
                ema = a if (ema is None or mode == "fallback-largest") else 0.5 * ema + 0.5 * a
                lost = 0
                # Drop-in-Vergleich zu v1 (argmax bbox_area ueber ALLE people)
                if people and idx != people.index(max(people, key=bbox_area)):
                    v1_diff += 1
                selection.append({"frame": i, "mode": mode, "person_idx": idx, "dist": round(dist, 1)})
            else:
                selection.append({"frame": i, "mode": "lost", "person_idx": None, "dist": None})

        # Gap-Interpolation (two-pass); Rand-Luecken <= max_gap werden per
        # Hold-Fill geschlossen statt abzubrechen (ein einziger schlechter
        # Randframe — Motion Blur am Clip-Anfang — vernichtete sonst den Lauf).
        interpolated = 0
        i = 0
        while i < len(frames):
            if chosen[i] is None:
                j = i
                while j < len(frames) and chosen[j] is None:
                    j += 1
                gap = j - i
                prev_ok = i - 1 >= 0 and chosen[i - 1] is not None
                next_ok = j < len(frames)
                if gap > args.max_gap or not (prev_ok or next_ok):
                    print(f"FEHLER: Luecke von {gap} Frames ab Frame {i} "
                          f"(max {args.max_gap}, prev={prev_ok}, next={next_ok}) — Abbruch.", file=sys.stderr)
                    sys.exit(2)
                if prev_ok and next_ok:
                    a_p, b_p = chosen[i - 1][0], chosen[j][0]
                    for k in range(i, j):
                        t = (k - i + 1) / (gap + 1)
                        chosen[k] = [lerp_person(a_p, b_p, t)]
                        selection[k] = {"frame": k, "mode": "interpolated", "person_idx": None, "dist": None}
                        interpolated += 1
                else:
                    # Rand-Luecke: naechstliegende gueltige Person halten
                    hold, mode = ((chosen[j][0], "hold-first") if next_ok
                                  else (chosen[i - 1][0], "hold-last"))
                    for k in range(i, j):
                        chosen[k] = [hold]
                        selection[k] = {"frame": k, "mode": mode, "person_idx": None, "dist": None}
                i = j
            else:
                i += 1

    # --- Synthetik-Gate (Audit 04.08. F5): VOR jedem Touch am alten Output.
    # Ohne Gate konnten bis zu ~89% der Frames erfunden sein (Interpolation +
    # Hold-Fill je unter --max-gap) und der Lauf endete trotzdem mit DONE/Exit 0.
    if not args.keep_all:
        synth = sum(1 for s in selection
                    if s["mode"] in ("interpolated", "hold-first", "hold-last"))
        if synth > args.max_synth_share * len(frames):
            print(f"FEHLER: {synth}/{len(frames)} Frames synthetisch "
                  f"(> {args.max_synth_share:.0%}, --max-synth-share) — Tracker-Ergebnis "
                  f"nicht vertrauenswuerdig, Abbruch. Alter Output bleibt unangetastet.",
                  file=sys.stderr)
            sys.exit(2)

    # --- Transaktionale Publikation (Audit 04.08. F4): Lock -> Temp-Render ->
    # Validierung -> alter Ordner faellt erst JETZT -> rename + atomarer Sidecar.
    # Vorher: Ordner wurde geleert, DANN gerendert — ein Fehlschlag mitten im
    # Rendern vernichtete den alten Satz und liess Teil-PNGs + stale Sidecar zurueck.
    outbase = args.outdir.rstrip("\\/")
    lock_path = outbase + ".lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"FEHLER: Lock {lock_path} existiert — paralleler Lauf auf dasselbe Ziel? "
              f"Sonst Lock-Datei manuell loeschen.", file=sys.stderr)
        sys.exit(3)
    tmpdir = outbase + f".tmp-{os.getpid()}"
    try:
        os.makedirs(tmpdir)
        # Rendern — Canvas mit Bottom-Pad; Pixelkoordinaten bleiben identisch,
        # weil decode und draw dieselben Canvas-Masse nutzen.
        for i, f in enumerate(frames):
            ff = {"people": chosen[i] or [], "canvas_width": W, "canvas_height": H_out}
            poses, _, h, w = decode_json_as_poses(ff)
            img = draw_poses(poses, h, w, True, True, True)
            Image.fromarray(img).save(os.path.join(tmpdir, f"pose_{i:04d}.png"))

        n_png = sum(1 for e in os.listdir(tmpdir) if e.lower().endswith(".png"))
        if n_png != len(frames):
            print(f"FEHLER: Temp-Render unvollstaendig ({n_png}/{len(frames)} PNGs) — "
                  f"nichts publiziert.", file=sys.stderr)
            sys.exit(3)

        modes = [s["mode"] for s in selection]
        meta = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "src": os.path.abspath(args.src),
            "frames_total": len(frames),
            "fps": args.fps,
            "canvas": [W, H_out],
            "pad_bottom": args.pad_bottom,
            "params": {"max_gap": args.max_gap, "max_jump": args.max_jump,
                       "lost_fallback": args.lost_fallback,
                       "max_synth_share": args.max_synth_share},
            "counts": {m: modes.count(m) for m in sorted(set(modes))},
            "v1_selection_diffs": (None if args.keep_all else v1_diff),
            "selection": selection,
        }
        meta_tmp = outbase + ".meta.json.tmp"
        with open(meta_tmp, "w") as fh:
            json.dump(meta, fh, indent=1)

        safe_remove_outdir(args.outdir)
        os.rename(tmpdir, args.outdir)
        # Sidecar NEBEN dem Ordner, atomar (Ordner bleibt reine PNG-Sequenz fuer VHS)
        os.replace(meta_tmp, outbase + ".meta.json")
    finally:
        if os.path.isdir(tmpdir):  # nur im Fehlerfall noch vorhanden
            shutil.rmtree(tmpdir, ignore_errors=True)
        os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    print(f"DONE: {len(frames)} Frames -> {args.outdir}")
    print(f"  Modi: {meta['counts']}")
    if not args.keep_all:
        print(f"  Abweichungen zur v1-Auswahl (argmax-area): {v1_diff}")


if __name__ == "__main__":
    main()
