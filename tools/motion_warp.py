#!/usr/bin/env python
"""BotterDancer — Motion-Time-Warp: Choreo-Akzente aufs Beat-Grid des Ziel-Tracks.

Choreographer-Stufe 1 (Produktentscheidung 04.07., Review-Fix #7-Reihenfolge:
Rotationen -> Time-Warp -> Kontakte -> IK — dieses Tool ist der Time-Warp-Schritt;
Kontakt-Rekonstruktion/Foot-IK sind die naechste Stufe und werden hier nur
gemessen, nicht korrigiert).

  python motion_warp.py --src hmr4d_results.pt --analyze
  python motion_warp.py --src hmr4d_results.pt --grid beat.wav.beats.json
                        --out hmr4d_results_warped.pt

--analyze: Eigentempo der Choreo drucken (+ kompatible Ziel-BPM-Spanne).
Warp:      monotone stueckweise-lineare Zeit-Abbildung (max. +/-15%% lokale
           Dehnung), SMPL-Rotationen per Quaternion-Slerp resampelt, transl
           linear. QA wird MITGELIEFERT: Beat-Alignment vor/nach (am neu
           berechneten SMPL-Forward des Warps gemessen, nicht angenommen)
           und eine Foot-Skate-Kennzahl als Vorher/Nachher-Vergleich.
Nur smpl_params_global wird gewarpt (reproject_camera nutzt genau das);
smpl_params_incam/net_outputs bleiben ORIGINAL und sind nach dem Warp
zeitlich inkonsistent — Sidecar dokumentiert das.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\GVHMR")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmr4d.utils.body_model.smplx_lite import SmplxLiteCoco17  # noqa: E402
from motion_warp_core import (  # noqa: E402
    axis_angle_to_quat,
    best_grid_offset,
    build_warp_map,
    estimate_tatum,
    find_accents,
    quat_to_axis_angle,
    select_strongest,
    slerp,
    warp_time,
)

L_ANKLE, R_ANKLE = 15, 16  # COCO17


def joints_of(params, device):
    model = SmplxLiteCoco17().to(device).eval()
    with torch.no_grad():
        return model(*(params[k].to(device) for k in
                       ("body_pose", "betas", "global_orient", "transl"))).cpu().numpy()


def speed_curve(j17, fps):
    v = np.linalg.norm(np.diff(j17, axis=0), axis=-1).mean(axis=-1) * fps  # (F-1,)
    return np.concatenate([v[:1], v])


def foot_skate(j17, fps, contact_h=0.05):
    """Mittlere horizontale Knoechel-Geschwindigkeit (m/s) in Bodenkontakt-Frames.
    Kennzahl, kein Fix — Foot-IK ist die naechste Choreographer-Stufe."""
    floor = j17[..., 1].min()
    vals = []
    for a in (L_ANKLE, R_ANKLE):
        h = j17[:, a, 1] - floor
        vxz = np.linalg.norm(np.diff(j17[:, a, :][:, [0, 2]], axis=0), axis=-1) * fps
        contact = h[:-1] < contact_h
        if contact.any():
            vals.append(float(vxz[contact].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def alignment_error(accents, beats):
    if len(accents) == 0:
        return float("nan")
    beats = np.asarray(beats)
    return float(np.mean([np.abs(beats - a).min() for a in accents]))


def main():
    ap = argparse.ArgumentParser(description="Motion-Time-Warp aufs Beat-Grid")
    ap.add_argument("--src", required=True, help="hmr4d_results.pt (GVHMR)")
    ap.add_argument("--out", help="Ziel-.pt (Pflicht ausser bei --analyze)")
    ap.add_argument("--grid", help="Beat-Grid-JSON (make_beat_track.py-Sidecar)")
    ap.add_argument("--analyze", action="store_true", help="nur Eigentempo drucken")
    ap.add_argument("--target-bpm", type=float,
                    help="bei --analyze: optimalen Grid-Offset fuer diese BPM drucken")
    ap.add_argument("--max-stretch", type=float, default=0.15)
    ap.add_argument("--anchors", type=int, default=8,
                    help="nur die N staerksten Akzente ankern (starke Hits treffen den Beat)")
    ap.add_argument("--tatum", type=int, default=2,
                    help="Anker-Raster = Beat-Unterteilung (2 = Achtel: Synkopen "
                         "duerfen auf Offbeats landen, wie echte Choreo)")
    ap.add_argument("--src-fps", type=float, default=30000.0 / 1001.0)
    args = ap.parse_args()
    if not args.analyze and not (args.out and args.grid):
        ap.error("--out und --grid sind Pflicht (oder --analyze).")

    pred = torch.load(args.src, weights_only=False)
    sp = pred["smpl_params_global"]
    n = sp["body_pose"].shape[0]
    fps = args.src_fps
    duration = n / fps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    j17 = joints_of(sp, device)
    speed = speed_curve(j17, fps)
    accents = find_accents(speed, fps)
    tatum, t_off, t_resid = estimate_tatum(accents)
    print(f"[warp] {n} Frames ({duration:.2f}s), {len(accents)} Bewegungs-Akzente; "
          f"Tatum {tatum:.3f}s @ Offset {t_off:.3f}s (Restfehler {t_resid * 1000:.0f}ms)")
    print(f"[warp] Tempo-Kandidaten: Tatum als Achtel -> {30.0 / tatum:.1f} BPM, "
          f"als Viertel -> {60.0 / tatum:.1f} BPM")

    if args.analyze:
        if args.target_bpm:
            p = 60.0 / args.target_bpm
            off, resid = best_grid_offset(accents, p)
            print(f"  optimaler Grid-Offset fuer {args.target_bpm:g} BPM (Viertel): {off:.3f}s "
                  f"(Restfehler {resid * 1000:.0f}ms vor Warp) -> "
                  f"make_beat_track.py --bpm {args.target_bpm:g} --offset {off:.3f}")
        return 0

    with open(args.grid, "r", encoding="utf-8-sig") as f:
        grid = json.load(f)
    # Anker-Raster: Beats des Tracks, unterteilt (--tatum 2 = Achtel). Echte
    # Choreo synkopiert — Hits auf Offbeats sind musikalisch richtig und
    # duerfen nicht auf den naechsten Viertel-Beat gezerrt werden.
    tatum_period = 60.0 / float(grid["bpm"]) / args.tatum
    beats = np.arange(float(grid.get("offset_s", 0.0)), duration + 0.25, tatum_period)
    strong = select_strongest(accents, speed, fps, args.anchors)
    err_before = alignment_error(strong, beats)
    skate_before = foot_skate(j17, fps)

    anchors, matched, dropped = build_warp_map(strong, duration, beats, args.max_stretch)
    print(f"[warp] Anker: {matched}/{len(strong)} starke Akzente aufs "
          f"{args.tatum}er-Tatum gezogen, {dropped} verworfen "
          f"(Steigungsschranke +/-{args.max_stretch:.0%})")
    if matched == 0:
        print("FEHLER: kein einziger Akzent innerhalb der Dehnungsgrenze — Ziel-BPM "
              "passt nicht zum Eigentempo (--analyze nutzen).", file=sys.stderr)
        return 2

    # Resample: Ausgabe behaelt Framezahl/fps; Zielzeit -> Quellzeit -> Interpolation
    t_out = np.arange(n) / fps
    t_src = warp_time(anchors, np.clip(t_out, 0, anchors[-1, 0]))
    f_idx = np.clip(t_src * fps, 0, n - 1)
    i0 = np.floor(f_idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    u = (f_idx - i0)[:, None]

    def lerp(x):
        x = x.cpu().numpy()
        return torch.from_numpy(x[i0] + (x[i1] - x[i0]) * u.reshape(-1, *([1] * (x.ndim - 1)))
                                ).float()

    def slerp_aa(x, joints):
        aa = x.cpu().numpy().reshape(n, joints, 3)
        q = slerp(axis_angle_to_quat(aa[i0]), axis_angle_to_quat(aa[i1]), u[:, 0])
        return torch.from_numpy(quat_to_axis_angle(q).reshape(n, joints * 3)).float()

    warped = {
        "body_pose": slerp_aa(sp["body_pose"], 21),
        "global_orient": slerp_aa(sp["global_orient"], 1),
        "betas": lerp(sp["betas"]),
        "transl": lerp(sp["transl"]),
    }
    pred["smpl_params_global"] = warped

    # QA am ECHTEN Ergebnis: SMPL-Forward des Warps, Akzente neu messen.
    # Anker-Praezision = pro gezogenem Beat der naechste NEU detektierte Akzent —
    # misst, ob die Hits wirklich gelandet sind (Mittel ueber alle Akzente wuerde
    # unerreichbare Neben-Minima einmischen und das Ergebnis verfaelschen).
    j17_w = joints_of(warped, device)
    accents_after = find_accents(speed_curve(j17_w, fps), fps)
    hit_beats = anchors[1:-1, 0] if anchors.shape[0] > 2 else np.empty(0)
    if len(hit_beats) and len(accents_after):
        anchor_err = float(np.mean([np.abs(np.asarray(accents_after) - b).min()
                                    for b in hit_beats]))
    else:
        anchor_err = float("nan")
    err_after = alignment_error(accents_after, beats)
    skate_after = foot_skate(j17_w, fps)
    print(f"[warp] Anker-Praezision nach Warp: {anchor_err * 1000:.0f}ms "
          f"(gezogene Hits vs. naechster redetektierter Akzent)")
    print(f"[warp] Beat-Alignment gesamt: {err_before * 1000:.0f}ms -> {err_after * 1000:.0f}ms "
          f"(inkl. bewusst nicht gezogener Akzente)")
    print(f"[warp] Foot-Skate-Kennzahl: {skate_before:.3f} -> {skate_after:.3f} m/s "
          f"(Kontakt-Frames; Fix = naechste Stufe Foot-IK)")

    torch.save(pred, args.out)
    meta = {
        "src": os.path.abspath(args.src), "grid": os.path.abspath(args.grid),
        "bpm": grid.get("bpm"), "max_stretch": args.max_stretch,
        "anchors_dst_src": anchors.tolist(), "matched": matched, "dropped": dropped,
        "alignment_ms": {"before": err_before * 1000, "after": err_after * 1000},
        "foot_skate_ms": {"before": skate_before, "after": skate_after},
        "warning": "smpl_params_incam/net_outputs sind NICHT gewarpt (original)",
    }
    with open(args.out + ".warp.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"DONE: {args.out} (+ .warp.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
