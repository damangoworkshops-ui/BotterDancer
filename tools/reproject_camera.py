"""BotterDancer Stage 2 — Kamera-Reprojektions-Bridge.

Nimmt hmr4d_results.pt (GVHMR, weltverankerte SMPL-Parameter) und rendert das
Skelett aus einer frei waehlbaren virtuellen Kamera als OpenPose-18-Stickfigur-
Videosequenz — im EXAKT selben Format/Rendering-Pfad, den unser bereits
validierter Wan-Animate-Workflow als pose_video konsumiert.

Design-Entscheidungen (verifiziert, nicht geraten):
- Joint-Ausgabe ueber hmr4d.utils.body_model.smplx_lite (SmplxLiteSmplN24 fuer
  die GVHMR-eigene Kanonisierung [compute_T_ayfz2ay erwartet exakt das
  SMPL-24-Schema: idx0=Becken, idx1/2=Huefte, idx16/17=Schulter], SmplxLiteCoco17
  fuer die tatsaechlichen Keypoints) statt volles SMPL-X-Mesh — schneller, und
  matcht render_global()'s eigene Konvention (Y-up, "face +Z" Kanonisierung).
- Projektion ueber hmr4d.utils.geo_transform.project_p2d (vorhandene, im Repo
  genutzte Pinhole-Projektion) statt eigener Neuerfindung.
- Rendering ueber custom_controlnet_aux.dwpose.decode_json_as_poses/draw_poses —
  IDENTISCH zum Pfad in filter_pose_v2.py, der im ISO-Test bereits nachweislich
  mit Wan-Animate funktioniert (kein Stil-/Dicke-Mismatch-Risiko).
- COCO17->OpenPose18-Mapping am Code verifiziert (wis3d_utils.py KINEMATIC_CHAINS
  "coco17"-Eintrag + limbSeq in dwpose/util.py), nicht aus dem Gedaechtnis geraten.
- Eigene Look-at-Kamera (Standardformel) statt GVHMRs get_global_cameras_static,
  weil letztere pytorch3d.renderer.cameras.look_at_rotation braucht — das ist in
  unserem Shim absichtlich NICHT vorhanden (nur transforms, kein compilierter
  Renderer) und wuerde crashen.

Nutzung (im GVHMR-venv):
  python reproject_camera.py --src <hmr4d_results.pt> --outdir <dir>
      --azimuth 90 --elevation 15 --distance-scale 1.5 --width 512 --height 768
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\GVHMR")
sys.path.insert(0, r"C:\ComfyUI\custom_nodes\comfyui_controlnet_aux\src")

from hmr4d.utils.body_model.smplx_lite import SmplxLiteCoco17, SmplxLiteSmplN24  # noqa: E402
from hmr4d.utils.geo_transform import (  # noqa: E402
    apply_T_on_points,
    compute_T_ayfz2ay,
    project_p2d,
)
from hmr4d.utils.geo.hmr_cam import create_camera_sensor  # noqa: E402
from custom_controlnet_aux.dwpose import decode_json_as_poses, draw_poses  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cam_trajectory import EASINGS, trajectory  # noqa: E402

# COCO17 (Standard-Reihenfolge, verifiziert via wis3d_utils.py KINEMATIC_CHAINS["coco17"]
# + hmr4d/utils/vis/cv2_utils.py) -> OpenPose-18 Index (verifiziert via dwpose/util.py
# limbSeq: 1=Nase,2=Nacken,3=RSchulter,4=RElbogen,5=RHandgelenk,6=LSchulter,7=LElbogen,
# 8=LHandgelenk,9=RHuefte,10=RKnie,11=RKnoechel,12=LHuefte,13=LKnie,14=LKnoechel,
# 15=RAuge,16=LAuge,17=ROhr,18=LOhr -- hier 0-indiziert).
# None = synthetischer Nacken (Mittelpunkt der Schultern), separat berechnet.
COCO17_TO_OPENPOSE18 = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
COCO_L_SHOULDER, COCO_R_SHOULDER = 5, 6


def look_at_R_t(eye, target, world_up=(0.0, 1.0, 0.0)):
    """Standard Look-at-Kamera (Y-up-Welt), OpenCV-Bildkonvention (X-rechts, Y-unten, Z-vorwaerts).
    Returns R_w2c (3,3), t_w2c (3,) mit X_cam = R_w2c @ X_world + t_w2c.

    KORRIGIERT (Review 2026-07-19): die alte Basis x=cross(up,z), y=cross(x,z)
    hatte det(R) = -1 — eine SPIEGELUNG, kein Rotation. Alle damit erzeugten
    Renders waren horizontal seitenverkehrt (Chiralitaet der Choreo invertiert);
    der alte Vier-Punkte-Test konnte das nicht sehen, weil ein konsistent
    gespiegeltes "rechts" von innen wie "rechts" aussieht. Rechtshaendig:
    x = cross(z, up), y = cross(z, x) -> det(R) = +1, y zeigt im Bild nach unten,
    Welt-+X landet bei azimuth 0 auf Bild-rechts.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)

    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)  # bereits normiert (z_axis, x_axis orthonormal)

    R_w2c = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3,3)
    det = float(np.linalg.det(R_w2c))
    assert det > 0.99, f"Kamera-Basis muss echte Rotation sein, det={det:.3f}"
    t_w2c = -R_w2c @ eye
    return R_w2c, t_w2c


def vhs_select_indices(n_src, src_fps, target_fps):
    """Bitgenaue Nachbildung der VHS-force_rate-Frameauswahl (cv_frame_generator,
    ComfyUI-VideoHelperSuite load_video_nodes.py, Akkumulator-Verfahren).
    NICHT durch eine Rundungsformel ersetzen: schon Index 8 weicht bei
    150@29.97->16 zwischen round() und dem Akkumulator ab — die Pose-Frames
    muessen exakt so ausgewaehlt werden wie VHS die Videoframes auswaehlt,
    sonst driften Pose und Referenz zeitlich auseinander."""
    if not target_fps:
        return list(range(n_src))
    if src_fps <= 0 or target_fps < 0:
        raise ValueError(f"ungueltige fps: src={src_fps}, target={target_fps}")
    base = 1.0 / src_fps        # base_frame_time
    target = 1.0 / target_fps   # target_frame_time
    acc = target                # time_offset startet bei target -> Frame 0 wird immer emittiert
    i = 0
    out = []
    while True:
        if acc < target:        # naechsten Quellframe "grabben"
            i += 1
            if i >= n_src:
                break
            acc += base
        if acc < target:
            continue
        acc -= target
        out.append(i)           # bei target < base wird i wiederholt (Upsampling-Duplikate)
    return out


# OpenPose-18 Indizes (0-indiziert) der Gesichtspunkte: Nase, RAuge, LAuge, ROhr, LOhr.
FACE_OPENPOSE18_IDXS = [0, 14, 15, 16, 17]


def build_openpose18_frame(coco17_xy, canvas_w, canvas_h, face_visible=True, valid=None):
    """coco17_xy: (17,2) pixel-Koordinaten -> OpenPose-18 pose_keypoints_2d (flach, 54 Werte).

    face_visible=False setzt Confidence der Gesichtspunkte (Nase/Augen/Ohren) auf 0.0 ->
    decode_json_as_poses().create_keypoint() macht daraus None -> draw_bodypose() ueberspringt
    das Liniensegment komplett (verifiziert in dwpose/util.py: `if c < 1.0: return None`,
    `if keypoint1 is None or keypoint2 is None: continue`). Kein Punkt bei (0,0), sauberes Weglassen.

    valid: optionale (17,)-Bool-Maske — False (hinter der Kamera / ausserhalb des
    Canvas) setzt die Confidence auf 0.0, statt Garbage-Staebe zu zeichnen.
    """
    kp18 = np.zeros((18, 3), dtype=np.float64)
    for op_idx, coco_idx in enumerate(COCO17_TO_OPENPOSE18):
        if coco_idx is None:
            neck = (coco17_xy[COCO_L_SHOULDER] + coco17_xy[COCO_R_SHOULDER]) / 2.0
            kp18[op_idx, :2] = neck
            ok = valid is None or bool(valid[COCO_L_SHOULDER] and valid[COCO_R_SHOULDER])
        else:
            kp18[op_idx, :2] = coco17_xy[coco_idx]
            ok = valid is None or bool(valid[coco_idx])
        is_face = op_idx in FACE_OPENPOSE18_IDXS
        kp18[op_idx, 2] = 1.0 if (ok and not (is_face and not face_visible)) else 0.0
    return kp18.reshape(-1).tolist()


def compute_facing_z(hips_l, hips_r, shoulders_l, shoulders_r):
    """Pro-Frame horizontale 'Blickrichtung' des Koerpers (Y-up-Welt), IDENTISCHE Formel wie
    hmr4d.utils.geo_transform.compute_T_ayfz2ay (dieselben Hueft-/Schulter-Indizes, dieselbe
    Kreuzprodukt-Reihenfolge) -- wiederverwendet statt neu erfunden, um KEINEN zweiten
    Vorzeichenfehler zu riskieren (siehe test_camera_axes.py-Vorfall bei der Kamera-Basis).
    Args: je (F,2) xz-Koordinaten. Returns: (F,2) normierte Facing-Vektoren in der xz-Ebene.
    """
    RL_xz = (hips_l - hips_r) + (shoulders_l - shoulders_r)  # (F,2), "leftward"
    x_dir = RL_xz / np.linalg.norm(RL_xz, axis=-1, keepdims=True).clip(min=1e-6)
    # z_dir = cross(x_dir, y=(0,1,0)) in 3D, auf die xz-Ebene reduziert.
    # Formel EMPIRISCH gegen hmr4d.utils.geo_transform.compute_T_ayfz2ay verifiziert
    # (scratchpad/test_facing_z.py) -- erste Version hatte hier ein Vorzeichen vertauscht.
    z_dir = np.stack([-x_dir[:, 1], x_dir[:, 0]], axis=-1)
    return z_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Pfad zu hmr4d_results.pt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--azimuth", type=float, default=90.0, help="Grad, 0=Original-Blick, 90=Seitlich")
    ap.add_argument("--elevation", type=float, default=10.0, help="Grad ueber der Horizontalen [-85..85]")
    ap.add_argument("--distance-scale", type=float, default=1.0,
                    help="Zusatzfaktor auf die FOV-basierte Mindestdistanz (1.0 = ganze Szene sicher im Bild)")
    # One-Shot-Trajektorien (04.08.): *-end macht den Parameter zum geeasten
    # Verlauf ueber die volle Sequenz -> virtuelle Steadicam/Crane statt Schnitt.
    # Ein One-Shot ist EINE Generation, d.h. Identitaets-Drift entfaellt
    # konstruktionsbedingt. Referenz-Sektor beachten: Front-Referenz traegt
    # ~0-120 Grad, Rueck-Referenz den Rest — Boegen pro Take, kein Vollkreis.
    ap.add_argument("--azimuth-end", type=float, default=None,
                    help="Azimut am Shot-Ende (aktiviert Kamerafahrt)")
    ap.add_argument("--elevation-end", type=float, default=None,
                    help="Elevation am Shot-Ende (Crane-Move)")
    ap.add_argument("--distance-scale-end", type=float, default=None,
                    help="Distanz-Skala am Shot-Ende (Push-in/Pull-out)")
    ap.add_argument("--easing", choices=list(EASINGS), default="smoothstep",
                    help="Verlaufsform der Kamerafahrt (smoothstep = ruckfrei anfahren/ausrollen)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--focal-mm", type=float, default=30.0)
    ap.add_argument("--target-fps", type=float, default=16.0,
                    help="Ziel-fps (VHS-force_rate-Semantik, wie save_pose_81.json); 0 = kein Resampling. "
                         "Ohne Resampling zeigten die cam-Renders nur ~54%% des Tanzes in Zeitlupe (Audit-Hauptbefund).")
    ap.add_argument("--src-fps", type=float, default=30000.0 / 1001.0,
                    help="fps des Quellvideos, auf dem GVHMR lief (Default: NTSC 29.97002997)")
    args = ap.parse_args()
    for label, val in (("--elevation", args.elevation), ("--elevation-end", args.elevation_end)):
        if val is not None and not (-85.0 <= val <= 85.0):
            ap.error(f"{label} {val} ausserhalb [-85, 85]: bei +/-90 ist die "
                     f"Look-at-Basis degeneriert (Nullvektor-Normierung -> NaN-Keypoints).")
    if os.path.lexists(args.outdir) and not os.path.isdir(args.outdir):
        # Frueh pruefen (z.B. Sidecar .meta.json als --outdir vertippt) — nicht erst
        # nach Minuten SMPL-Forward mit rohem Traceback sterben.
        ap.error(f"--outdir ist eine DATEI, kein Ordner: {args.outdir}")

    # --- 1. Motion-Daten laden ---
    pred = torch.load(args.src, weights_only=False)
    sp = pred["smpl_params_global"]
    body_pose, betas, global_orient, transl = sp["body_pose"], sp["betas"], sp["global_orient"], sp["transl"]
    n_frames = body_pose.shape[0]
    print(f"[reproject] {n_frames} Frames geladen aus {args.src}")
    # Frame-Auswahl frueh berechnen: Clipping-Statistik und Rendern beziehen sich
    # auf die GERENDERTEN Frames (Re-Attack: meta durfte nicht Quell-Frames zaehlen).
    sel = vhs_select_indices(n_frames, args.src_fps, args.target_fps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl24 = SmplxLiteSmplN24().to(device).eval()
    coco17 = SmplxLiteCoco17().to(device).eval()

    with torch.no_grad():
        bp, be, go, tr = (x.to(device) for x in (body_pose, betas, global_orient, transl))
        j24 = smpl24(bp, be, go, tr)  # (F, 24, 3)
        j17 = coco17(bp, be, go, tr)  # (F, 17, 3)

    # --- 2. Kanonisierung: identisch zu render_global()'s move_to_start_point_face_z,
    #        aber auf Gelenken statt Vertices (wir brauchen kein Mesh). ---
    floor_y = j24[..., 1].min()  # Approximation: tiefstes Gelenk statt tiefster Vertex
    offset = j24[0, 0, :].clone()  # Becken bei Frame 0
    offset[1] = floor_y
    j24 = j24 - offset
    j17 = j17 - offset
    T_ay2ayfz = compute_T_ayfz2ay(j24[[0]], inverse=True)  # (1,4,4), aus Frame-0-Facing
    j24 = apply_T_on_points(j24, T_ay2ayfz.expand(n_frames, -1, -1))
    j17 = apply_T_on_points(j17, T_ay2ayfz.expand(n_frames, -1, -1))

    j17_np = j17.cpu().numpy()  # (F, 17, 3), Y-up, Start bei Frame0 nahe Ursprung, Blick +Z

    # --- 3. Virtuelle Kamera: Orbit um die Szenen-Mitte ---
    # Distanz FOV-bewusst (Review 2026-07-19): der alte blinde Faktor 1.6x auf
    # extent liess laterale Bewegung aus dem Canvas laufen. Kugel-Schranke mit
    # sin (nicht tan: Tangentialpunkte erscheinen unter groesserem Winkel als
    # Punkte der Zentralebene; tan waere erst mit margin >= 1/cos(hfov) sicher).
    _, _, K = create_camera_sensor(args.width, args.height, args.focal_mm)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    K = K.to(device)

    center = j17_np.reshape(-1, 3).mean(axis=0)
    extent = np.linalg.norm(j17_np.reshape(-1, 3) - center, axis=-1).max()
    half_fov_min = np.arctan(min(cx / fx, cy / fy))

    # Trajektorien ueber die VOLLE Quell-Sequenz (Resampling waehlt danach dieselben
    # Frames wie fuers Pose-Rendern -> Kamera und Motion bleiben exakt synchron).
    # Ohne *-end-Argumente sind alle Verlaeufe konstant = exakt das Altverhalten.
    az_deg = trajectory(n_frames, args.azimuth, args.azimuth_end, args.easing)
    el_deg = trajectory(n_frames, args.elevation, args.elevation_end, args.easing)
    dscale = trajectory(n_frames, args.distance_scale, args.distance_scale_end, args.easing)
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    base_dist = extent / np.sin(half_fov_min) * 1.05
    distance = np.maximum(base_dist * dscale, 1.0)  # (F,)

    eye = center[None, :] + distance[:, None] * np.stack(
        [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], axis=-1
    )  # (F, 3)
    R_stack = np.empty((n_frames, 3, 3))
    t_stack = np.empty((n_frames, 3))
    for f in range(n_frames):
        R_stack[f], t_stack[f] = look_at_R_t(eye[f], center)
    R_all = torch.from_numpy(R_stack).float().to(device)
    t_all = torch.from_numpy(t_stack).float().to(device)

    j17_cam = torch.einsum("fij,fkj->fki", R_all, j17) + t_all[:, None, :]  # (F, 17, 3)
    p2d = project_p2d(j17_cam, K.expand(n_frames, -1, -1))  # (F, 17, 2), Pixel-Koordinaten
    p2d_np = p2d.cpu().numpy()

    # Keypoints hinter der Kamera oder ausserhalb des Canvas -> Confidence 0
    # (statt Garbage-Staebe zu zeichnen); Neck erbt die Schulter-Validitaet.
    z_cam = j17_cam[..., 2].cpu().numpy()  # (F, 17)
    kp_valid = ((z_cam > 0.05)
                & (p2d_np[..., 0] >= 0) & (p2d_np[..., 0] < args.width)
                & (p2d_np[..., 1] >= 0) & (p2d_np[..., 1] < args.height))
    n_clipped = int((~kp_valid[sel]).sum())  # nur gerenderte Frames zaehlen
    if n_clipped:
        print(f"[reproject] WARNUNG: {n_clipped} Keypoints hinter Kamera/ausserhalb Canvas "
              f"-> Confidence 0 (ggf. --distance-scale erhoehen)", file=sys.stderr)

    # Sichtbarkeits-Gate (Audit 04.08. F6): eine voll geclippte Sequenz (zu kleine
    # Distanz, kaputte Trajektorie) erzeugte bisher schwarze Poseframes + DONE —
    # und kostete downstream einen kompletten Wan-Render ohne Conditioning.
    vis_per_frame = kp_valid[sel].sum(axis=1)
    n_low_vis = int((vis_per_frame < 4).sum())
    if n_low_vis > 0.2 * len(sel):
        print(f"FEHLER: {n_low_vis}/{len(sel)} Frames mit <4 sichtbaren Joints — "
              f"Kamera-Distanz/Trajektorie pruefen (--distance-scale erhoehen). "
              f"Nichts geschrieben, alter Output bleibt.", file=sys.stderr)
        sys.exit(4)

    moving = any(v is not None for v in (args.azimuth_end, args.elevation_end,
                                         args.distance_scale_end))
    if moving:
        print(f"[reproject] Kamerafahrt ({args.easing}): azimuth {az_deg[0]:.0f}->{az_deg[-1]:.0f} "
              f"elevation {el_deg[0]:.0f}->{el_deg[-1]:.0f} "
              f"distance {distance[0]:.2f}->{distance[-1]:.2f}m "
              f"(extent {extent:.2f}m, hfov {np.degrees(half_fov_min):.1f} Grad)")
    else:
        print(f"[reproject] Kamera: azimuth={args.azimuth} elevation={args.elevation} "
              f"distance={distance[0]:.2f}m (extent {extent:.2f}m, hfov "
              f"{np.degrees(half_fov_min):.1f} Grad) eye={eye[0].round(2).tolist()}")

    # --- 3b. Pro-Frame Gesichts-Sichtbarkeit: Blickrichtung des Koerpers vs. Kamera-Richtung.
    #     Loest den 180 Grad-Bug (Gesichts-Keypoints wurden bisher IMMER gezeichnet, auch bei
    #     Rueckenlage zur Kamera -> Wan-Animate rendert dann reflexhaft die Vorderansicht). ---
    j24_np = j24.cpu().numpy()
    hips_l, hips_r = j24_np[:, 1, [0, 2]], j24_np[:, 2, [0, 2]]
    shoulders_l, shoulders_r = j24_np[:, 16, [0, 2]], j24_np[:, 17, [0, 2]]
    facing_z = compute_facing_z(hips_l, hips_r, shoulders_l, shoulders_r)  # (F,2) in (x,z)
    pelvis_xz = j24_np[:, 0, [0, 2]]  # (F,2)
    eye_xz = eye[:, [0, 2]]  # (F,2) — bei Kamerafahrt pro Frame verschieden
    view_dir = eye_xz - pelvis_xz  # (F,2), Charakter -> Kamera
    view_dir = view_dir / np.linalg.norm(view_dir, axis=-1, keepdims=True).clip(min=1e-6)
    face_dot = (facing_z * view_dir).sum(axis=-1)  # (F,), >0 = Gesicht zeigt zur Kamera
    rl0 = float(np.linalg.norm((hips_l[0] - hips_r[0]) + (shoulders_l[0] - shoulders_r[0])))
    if rl0 < 0.05:
        print(f"[reproject] WARNUNG: Links-Rechts-Achse in Frame 0 fast degeneriert "
              f"(|RL|={rl0:.3f} m) — Frame-0-Kanonisierung und damit der azimuth-Bezug "
              f"koennen unzuverlaessig sein.", file=sys.stderr)
    # Boxfilter (3) + Hysterese statt Einzelschwelle: eine feste 0.1-Marge
    # verschiebt das Flackern nur, sie beseitigt es nicht (Seitenansicht-Shimmy).
    padded = np.concatenate([face_dot[:1], face_dot, face_dot[-1:]])
    face_dot_s = np.convolve(padded, np.ones(3) / 3.0, mode="valid")  # (F,)
    face_visible = np.zeros(n_frames, dtype=bool)
    vis = bool(face_dot_s[0] > 0.1)
    for fi, d in enumerate(face_dot_s):
        if vis and d < 0.05:
            vis = False
        elif not vis and d > 0.15:
            vis = True
        face_visible[fi] = vis
    print(f"[reproject] Gesicht sichtbar in {int(face_visible.sum())}/{n_frames} Frames "
          f"(dot min={face_dot.min():.2f} max={face_dot.max():.2f}, Hysterese 0.05/0.15)")

    # --- 4. fps-Resampling + OpenPose-18-JSON pro Frame bauen ---
    # Kanonisierung/Kamera/Projektion liefen auf der VOLLEN Sequenz (bitidentisch
    # zu "VHS subsampled das fertige Video"); erst die Frame-AUSWAHL resampelt.
    if len(sel) != n_frames:
        print(f"[reproject] Resample {args.src_fps:.5f} -> {args.target_fps} fps: "
              f"{n_frames} -> {len(sel)} Frames (VHS-force_rate-Semantik)")
    frames_json = []
    for src_i in sel:
        kp = build_openpose18_frame(p2d_np[src_i], args.width, args.height,
                                    face_visible=bool(face_visible[src_i]),
                                    valid=kp_valid[src_i])
        frames_json.append({
            "people": [{"pose_keypoints_2d": kp}],
            "canvas_width": args.width,
            "canvas_height": args.height,
        })

    # --- 5. Transaktionale Publikation (Audit 04.08. F4): Lock -> Temp-Render ->
    # Validierung -> alter Ordner faellt erst JETZT -> rename + atomare Sidecars.
    # Vorher: rmtree VOR dem Rendern — ein Fehlschlag mittendrin vernichtete den
    # alten Satz und liess Teil-PNGs + stale Sidecars zurueck.
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
        for i, f in enumerate(frames_json):
            poses, _, h, w = decode_json_as_poses(f)
            img = draw_poses(poses, h, w, True, False, False)  # nur Body (kein Face/Hand-Data)
            Image.fromarray(img).save(os.path.join(tmpdir, f"pose_{i:04d}.png"))

        n_png = sum(1 for e in os.listdir(tmpdir) if e.lower().endswith(".png"))
        if n_png != len(frames_json):
            print(f"FEHLER: Temp-Render unvollstaendig ({n_png}/{len(frames_json)} PNGs) — "
                  f"nichts publiziert.", file=sys.stderr)
            sys.exit(3)

        json_tmp = outbase + ".json.tmp"
        with open(json_tmp, "w") as f:
            json.dump(frames_json, f)
        meta = {
            "src": os.path.abspath(args.src),
            "n_src_frames": n_frames,
            "n_frames": len(sel),
            "resample": {"src_fps": args.src_fps, "target_fps": args.target_fps,
                         "mode": "vhs-force-rate-accumulator"},
            "camera": {"azimuth": [float(az_deg[0]), float(az_deg[-1])],
                       "elevation": [float(el_deg[0]), float(el_deg[-1])],
                       "distance": [float(distance[0]), float(distance[-1])],
                       "distance_scale": [float(dscale[0]), float(dscale[-1])],
                       "easing": args.easing, "moving": bool(moving),
                       "eye_start": eye[0].tolist(), "eye_end": eye[-1].tolist(),
                       "center": center.tolist(),
                       "basis": "right-handed det+1 (Spiegelungs-Fix 2026-07-19)"},
            "canvas": [args.width, args.height],
            "clipped_keypoints": n_clipped,
            "frames_low_visibility": n_low_vis,
            "face_visible_frames": int(face_visible[sel].sum()),
        }
        meta_tmp = outbase + ".meta.json.tmp"
        with open(meta_tmp, "w") as f:
            json.dump(meta, f, indent=1)

        # Publikation: alten Ordner validiert entfernen (rmtree-Guard, Audit P0-3)
        if os.path.lexists(args.outdir) and not os.path.isdir(args.outdir):
            print(f"FEHLER: --outdir ist eine DATEI, kein Ordner: {args.outdir}", file=sys.stderr)
            sys.exit(3)
        if os.path.isdir(args.outdir):
            foreign = [e for e in os.listdir(args.outdir)
                       if not (e.startswith("pose_") and e.lower().endswith(".png"))]
            if foreign:
                print(f"FEHLER: {args.outdir} enthaelt {len(foreign)} fremde Eintraege "
                      f"(z.B. {foreign[:3]}) — Abbruch statt rmtree.", file=sys.stderr)
                sys.exit(3)
            shutil.rmtree(args.outdir)
        os.rename(tmpdir, args.outdir)
        os.replace(json_tmp, outbase + ".json")
        os.replace(meta_tmp, outbase + ".meta.json")
    finally:
        if os.path.isdir(tmpdir):  # nur im Fehlerfall noch vorhanden
            shutil.rmtree(tmpdir, ignore_errors=True)
        os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    print(f"DONE: {len(sel)} Frames gerendert -> {args.outdir}")


if __name__ == "__main__":
    main()
