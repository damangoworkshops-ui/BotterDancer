#!/usr/bin/env python
"""BotterDancer — Chain-Workflow-Generator: lange Routinen als Chunk-Kette.

WanAnimateToVideo ist fuers Chunk-Chaining gebaut (video_frame_offset ist
Input UND Output, Tooltip: 'Connect to the video_frame_offset output of the
previous node'; continue_motion nimmt die Bilder des Vorgaenger-Chunks,
continue_motion_max_frames Kontextframes werden fortgesetzt). Dieser Generator
dupliziert die Chunk-Stufe (WanAnimate -> KSampler -> Trim -> Decode) N-fach,
verkettet offset+continue_motion und haengt alle Bilder per ImageBatch an
EINEN VHS_VideoCombine — Referenz/Prompts/Loader bleiben geteilt (Review-Fix
#5: jeder Chunk conditioned auf DIESELBE Referenz).

  python make_chain_workflow.py --base workflows/wan_animate_final_v2.json
      --chunks 2 --pose-dir C:\\ComfyUI\\input\\botter_pose_chain
      --pose-frames 160 --prefix chain_final --out workflows/chain_final_2x.json

Der Graph existiert weiterhin nur EINMAL als Basis; Chains werden generiert
(kein Kopien-Drift, Audit-Lehre P1-9).
"""
import argparse
import copy
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="Chunk-Chain-Workflow generieren")
    ap.add_argument("--base", required=True, help="Basis-Graph (wan_animate_final_v2/draft)")
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--chunk-length", type=int, default=81, help="Frames pro Chunk (step 4 +1)")
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--pose-frames", type=int, required=True,
                    help="Gesamtzahl PNGs im Pose-Verzeichnis (image_load_cap)")
    ap.add_argument("--prefix", required=True, help="filename_prefix des Outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8-sig") as f:
        doc = json.load(f)
    prompt = doc.get("prompt", doc)

    wan = [k for k, n in prompt.items() if n.get("class_type") == "WanAnimateToVideo"]
    ks = [k for k, n in prompt.items() if n.get("class_type") == "KSampler"]
    trim = [k for k, n in prompt.items() if n.get("class_type") == "TrimVideoLatent"]
    dec = [k for k, n in prompt.items() if n.get("class_type") == "VAEDecode"]
    comb = [k for k, n in prompt.items() if n.get("class_type") == "VHS_VideoCombine"]
    pose = [k for k, n in prompt.items() if n.get("class_type") == "VHS_LoadImagesPath"]
    if not all(len(x) == 1 for x in (wan, ks, trim, dec, comb, pose)):
        print("FEHLER: Basis-Graph muss genau 1x Wan/KSampler/Trim/Decode/Combine/"
              "LoadImagesPath enthalten.", file=sys.stderr)
        return 2
    wan, ks, trim, dec, comb, pose = wan[0], ks[0], trim[0], dec[0], comb[0], pose[0]

    prompt[pose]["inputs"]["directory"] = args.pose_dir
    prompt[pose]["inputs"]["image_load_cap"] = args.pose_frames
    prompt[wan]["inputs"]["length"] = args.chunk_length
    prompt[comb]["inputs"]["filename_prefix"] = args.prefix

    chunk_ids = [(wan, ks, trim, dec)]
    for c in range(1, args.chunks):
        ids = {}
        for base_id in (wan, ks, trim, dec):
            new_id = f"{base_id}{c:02d}"  # eindeutig, basiert auf Original-ID
            node = copy.deepcopy(prompt[base_id])
            prompt[new_id] = node
            ids[base_id] = new_id
        # interne Verdrahtung des Chunks auf die eigenen Kopien umbiegen
        for base_id in (ks, trim, dec):
            for iname, val in prompt[ids[base_id]]["inputs"].items():
                if isinstance(val, list) and len(val) == 2 and val[0] in ids:
                    prompt[ids[base_id]]["inputs"][iname] = [ids[val[0]], val[1]]
        prev_wan, prev_dec = chunk_ids[-1][0], chunk_ids[-1][3]
        w = prompt[ids[wan]]["inputs"]
        w["video_frame_offset"] = [prev_wan, 5]   # Output 5 = video_frame_offset
        w["continue_motion"] = [prev_dec, 0]      # Bilder des Vorgaenger-Chunks
        chunk_ids.append((ids[wan], ids[ks], ids[trim], ids[dec]))

    # Alle Chunk-Bilder in Reihenfolge batchen -> ein VideoCombine
    images = [dec_id for _, _, _, dec_id in chunk_ids]
    src = [images[0], 0]
    for i, dec_id in enumerate(images[1:], start=1):
        bid = f"batch{i:02d}"
        prompt[bid] = {"class_type": "ImageBatch",
                       "inputs": {"image1": src, "image2": [dec_id, 0]}}
        src = [bid, 0]
    prompt[comb]["inputs"]["images"] = src

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt}, f, indent=1)
    print(f"DONE: {args.out} ({args.chunks} Chunks x {args.chunk_length} Frames, "
          f"Pose-Sequenz {args.pose_frames} Frames, {len(prompt)} Nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
