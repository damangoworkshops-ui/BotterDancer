#!/usr/bin/env python
"""BotterDancer — Root-Anchoring gegen Foot-Skate (Choreographer-Stufe 2).

Nimmt ein (typischerweise bereits Time-gewarptes) hmr4d_results.pt, erkennt
Bodenkontakte der Knoechel und korrigiert die horizontale Wurzel-Trajektorie
(nur transl — keine Gelenkwinkel, kein Naturalness-Risiko), sodass Standfuesse
planted bleiben. QA am neu berechneten SMPL-Forward: Skate-Metrik vorher/nachher
+ maximale Pfadkorrektur. Restfehler in den Beinen selbst waere Stufe 3
(Zwei-Knochen-IK) — erst bauen, wenn die Metrik es nach dieser Stufe verlangt.

  python foot_anchor.py --src hmr4d_results_warped90.pt --out ..._anchored.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\GVHMR")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_warp import foot_skate, joints_of  # noqa: E402
from foot_core import (  # noqa: E402
    clean_contacts,
    detect_contacts,
    root_anchor_correction,
    skate_metric,
)

L_ANKLE, R_ANKLE = 15, 16  # COCO17


def feet_data(j17, fps):
    feet_xz = j17[:, [L_ANKLE, R_ANKLE]][:, :, [0, 2]]  # (F, 2, 2)
    feet_y = j17[:, [L_ANKLE, R_ANKLE], 1]              # (F, 2)
    v = np.linalg.norm(np.diff(feet_xz, axis=0), axis=-1) * fps
    feet_v = np.concatenate([v[:1], v], axis=0)          # (F, 2)
    return feet_xz, feet_y, feet_v


def main():
    ap = argparse.ArgumentParser(description="Root-Anchoring gegen Foot-Skate")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h-thresh", type=float, default=0.06, help="Kontakt-Hoehe (m ueber Boden)")
    ap.add_argument("--v-thresh", type=float, default=0.35, help="Kontakt-Maximalgeschwindigkeit (m/s)")
    ap.add_argument("--min-contact", type=int, default=3, help="Mindest-Kontaktlaenge (Frames)")
    ap.add_argument("--src-fps", type=float, default=30000.0 / 1001.0)
    args = ap.parse_args()

    pred = torch.load(args.src, weights_only=False)
    sp = pred["smpl_params_global"]
    fps = args.src_fps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    j17 = joints_of(sp, device)
    feet_xz, feet_y, feet_v = feet_data(j17, fps)
    floor = float(feet_y.min())
    contacts = np.stack([
        clean_contacts(detect_contacts(feet_y[:, f], feet_v[:, f], floor,
                                       args.h_thresh, args.v_thresh), args.min_contact)
        for f in range(2)], axis=1)
    n_contact = int(contacts.any(axis=1).sum())
    print(f"[anchor] {len(j17)} Frames, Kontakt in {n_contact} Frames "
          f"(L {int(contacts[:, 0].sum())}, R {int(contacts[:, 1].sum())})")
    if n_contact == 0:
        print("FEHLER: keine Kontakte erkannt — Schwellen pruefen "
              "(--h-thresh/--v-thresh).", file=sys.stderr)
        return 2

    skate_before = skate_metric(feet_xz, contacts, fps)
    ungated_before = foot_skate(j17, fps)
    corr = root_anchor_correction(feet_xz, contacts, fps)
    print(f"[anchor] max. Pfadkorrektur: {np.linalg.norm(corr, axis=-1).max() * 100:.1f} cm")

    # .copy(): sonst mutiert die Kette View->inplace->from_numpy den geladenen
    # Original-Tensor in pred (Verifier 05.08.: heute alias-frei, aber die
    # Korrektheit hing an einem unerzwungenen unsichtbaren Invariant).
    transl = sp["transl"].cpu().numpy().copy()
    transl[:, 0] += corr[:, 0]
    transl[:, 2] += corr[:, 1]
    sp["transl"] = torch.from_numpy(transl).float()
    pred["smpl_params_global"] = sp

    # QA am ECHTEN Ergebnis (nicht an der Annahme): Forward neu, beide Metriken —
    # gated (Kontakt-Maske dieses Tools) UND ungated (motion_warp-kompatibel,
    # hoehen-only): die Stufe-3-Entscheidung braucht stufenuebergreifend
    # vergleichbare Zahlen (Verifier-Befund 05.08.: 0.127 vs 0.276 auf derselben
    # Datei waren Aepfel/Birnen nebeneinander in zwei Sidecars).
    j17_a = joints_of(sp, device)
    feet_xz_a, _, _ = feet_data(j17_a, fps)
    skate_after = skate_metric(feet_xz_a, contacts, fps)
    ungated_after = foot_skate(j17_a, fps)
    print(f"[anchor] Foot-Skate (gated):   {skate_before:.3f} -> {skate_after:.3f} m/s "
          f"({(1 - skate_after / skate_before) * 100:+.0f}%)")
    print(f"[anchor] Foot-Skate (ungated): {ungated_before:.3f} -> {ungated_after:.3f} m/s "
          f"({(1 - ungated_after / ungated_before) * 100:+.0f}%, motion_warp-kompatibel)")

    torch.save(pred, args.out)
    meta = {
        "src": os.path.abspath(args.src),
        "params": {"h_thresh": args.h_thresh, "v_thresh": args.v_thresh,
                   "min_contact": args.min_contact},
        "contact_frames": {"any": n_contact, "left": int(contacts[:, 0].sum()),
                           "right": int(contacts[:, 1].sum())},
        "skate_ms_gated": {"before": skate_before, "after": skate_after},
        "skate_ms_ungated": {"before": ungated_before, "after": ungated_after},
        "max_path_correction_m": float(np.linalg.norm(corr, axis=-1).max()),
        "warning": "smpl_params_incam/net_outputs sind NICHT gewarpt/geankert (original)",
    }
    with open(args.out + ".anchor.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"DONE: {args.out} (+ .anchor.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
