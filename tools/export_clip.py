#!/usr/bin/env python
# BotterDancer -- terminaler Export-Schritt (Architektur-Review Fix #6), v2 nach
# Doppel-Audit 2026-08-04 (Codex extern + eigener Re-Attack-Pass, Findings gemerged).
#
# Ablauf (Reihenfolge ist Pflicht):
#   1. Preflight: Argumente, Tools, Signatur-Material, Eingabevertrag (ffprobe),
#      Decode-Integritaet -- ALLES bevor teure Pixelarbeit beginnt.
#   2. Pixel-Ops im Temp: unsichtbares Watermark + Re-Encode + Metadaten-Strip
#      (oder verlustfreier Remux bei --no-watermark).
#   3. C2PA-Signatur im Temp.
#   4. VOLLSTAENDIGE Verifikation im Temp (Manifest-Policy, Signer-Pinning,
#      Watermark an 3 Frames, Metadaten-Sweep ueber Format/Streams/Chapters).
#   5. Erst nach bestandener Pruefung atomare Publikation nach --outdir.
#      => Im Export-Ordner liegt NIE eine ungepruefte Datei.
#
# Hinweis zur C2PA-Bindung: c2pa.hash.bmff.v3 schuetzt die relevanten Boxen,
# NICHT jedes Byte der Datei (ftyp/mfra sind spezifikationsgemaess ausgenommen).
#
# Exit-Codes: 0=ok, 2=Eingabe/Argumente/Schluessel/Vertrag, 3=Encode/Integritaet,
#             4=Signatur-Fehler, 5=Verifikation fehlgeschlagen.
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

FFMPEG_FALLBACKS = [
    r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin",
]
KEYS_DIR = Path(__file__).resolve().parent.parent / "assets" / "keys"
GENERATOR = {"name": "BotterDancer", "version": "0.1"}
IPTC_TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
ALLOWED_FORMAT_TAGS = {"major_brand", "minor_version", "compatible_brands", "encoder"}
ALLOWED_STREAM_TAGS = {"language", "handler_name", "vendor_id", "encoder", "creation_time"}
MAX_TAG_VALUE_LEN = 64  # Prompt-Schmuggel in erlaubten Tags (z.B. handler_name) fangen


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for d in FFMPEG_FALLBACKS:
        cand = Path(d) / (name + ".exe")
        if cand.exists():
            return str(cand)
    return None


def fail(code, msg):
    print("FEHLER: " + msg)
    sys.exit(code)


def run(cmd, fail_code=3, what=""):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        tail = (r.stderr or b"").decode("utf-8", "replace")[-800:]
        fail(fail_code, f"{what or Path(cmd[0]).name} fehlgeschlagen (Exit {r.returncode}):\n{tail}")
    return r


# ---------------------------------------------------------------- Watermark-Fix
def _patch_imwatermark():
    """Upstream-Bug in invisible-watermark 0.2.0 (maxDct.py Z.30): dwt2 liefert
    (h1,v1,d1), idwt2 wird aber mit (v1,h1,d1) aufgerufen -- Detailbaender
    vertauscht. Auf unserem Pastell-Content gemessen folgenlos (PSNR 40,3 dB
    identisch), aber fuer detailreichen Content (Room-Keeper) falsch. Zusaetzlich
    clippt Upstream nicht vor dem uint8-Cast (Wrap-Around moeglich)."""
    import cv2
    import numpy as np
    import pywt
    from imwatermark.maxDct import EmbedMaxDct

    def encode_fixed(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue
            ca1, (h1, v1, d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
            self.encode_frame(ca1, self._scales[channel])
            rec = pywt.idwt2((ca1, (h1, v1, d1)), "haar")
            yuv[: row // 4 * 4, : col // 4 * 4, channel] = np.clip(rec, 0, 255)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    EmbedMaxDct.encode = encode_fixed


# ------------------------------------------------------------------- Preflight
def probe(ffprobe, path):
    r = run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-show_chapters",
             "-of", "json", str(path)], fail_code=3, what="ffprobe")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        fail(3, f"ffprobe lieferte kein JSON fuer {path}.")


def enforce_contract(info, path):
    """Eingabevertrag: genau EIN 8-bit-SDR-Videostream, CFR, gerade Masse,
    keine Rotation, keine Kapitel. Alles andere fail-closed statt still normalisieren."""
    streams = info.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        kinds = [s.get("codec_type") for s in streams]
        fail(2, f"Eingabevertrag verletzt: erwartet genau 1 Videostream, gefunden {kinds}. "
                "(Audio wuerde still verworfen -- bei Bedarf Export-Tool erweitern.)")
    v = streams[0]
    if v.get("pix_fmt") not in ("yuv420p", "yuvj420p"):
        fail(2, f"Eingabevertrag: pix_fmt {v.get('pix_fmt')} nicht unterstuetzt (nur 8-bit 4:2:0).")
    w, h = int(v["width"]), int(v["height"])
    if w % 2 or h % 2:
        fail(2, f"Eingabevertrag: ungerade Dimensionen {w}x{h}.")
    for sd in v.get("side_data_list", []):
        if "rotation" in sd or sd.get("side_data_type") == "Display Matrix":
            fail(2, "Eingabevertrag: Rotations-Metadaten vorhanden -- erst ausbacken.")
    rfr, afr = v.get("r_frame_rate", "0/0"), v.get("avg_frame_rate", "0/0")
    if rfr != afr or Fraction(rfr) <= 0:
        fail(2, f"Eingabevertrag: kein sauberes CFR (r={rfr}, avg={afr}).")
    if info.get("chapters"):
        fail(2, "Eingabevertrag: Kapitel im Container.")
    nframes = v.get("nb_frames")
    if not nframes or not str(nframes).isdigit() or int(nframes) < 1:
        fail(2, f"Eingabevertrag: nb_frames unbrauchbar ({nframes!r}).")
    return w, h, rfr, int(nframes)


def integrity_check(ffmpeg, path):
    """Vollstaendiger Decode mit hartem Fehlerabbruch: faengt beschaedigte GOPs,
    die OpenCV als stilles Vorzeiten-EOF maskieren wuerde."""
    run([ffmpeg, "-v", "error", "-xerror", "-err_detect", "explode",
         "-i", str(path), "-f", "null", "-"], fail_code=3, what="Integritaets-Decode")


def load_credentials():
    """Signatur-Material VOR jeder Pixelarbeit laden und validieren.
    Liefert (Signer, Leaf-Seriennummer als str) -- Seriennummer wird beim
    Verify gegen signature_info gepinnt."""
    from c2pa import C2paSignerInfo, C2paSigningAlg, Signer
    from cryptography import x509

    chain_p, key_p = KEYS_DIR / "c2pa_chain.pem", KEYS_DIR / "c2pa_leaf.key"
    if not chain_p.exists() or not key_p.exists():
        fail(2, f"Signatur-Schluessel fehlen in {KEYS_DIR} -- erst tools\\make_c2pa_cert.py ausfuehren.")
    chain = chain_p.read_bytes()
    try:
        leaf_serial = str(x509.load_pem_x509_certificates(chain)[0].serial_number)
    except Exception as e:
        fail(2, f"c2pa_chain.pem nicht parsebar: {e}")
    info = C2paSignerInfo(alg=C2paSigningAlg.ES256, sign_cert=chain,
                          private_key=key_p.read_bytes(), ta_url=b"")
    # Kein Timestamp-Server: ta_url muss im ctypes-Struct NULL sein. Der
    # Python-Konstruktor verbietet None, ein Leerstring laesst die native
    # Lib mit "Signature: empty string" sterben (empirisch isoliert 2026-08-04).
    info.ta_url = None
    try:
        signer = Signer.from_info(info)
    except Exception as e:
        fail(2, f"Signer aus Schluesselmaterial nicht erzeugbar: {e}")
    return signer, leaf_serial


# ------------------------------------------------------------------- Pixel-Ops
def watermark_and_reencode(ffmpeg, src, dst, payload, crf, w, h, fps, nframes):
    import cv2
    from imwatermark import WatermarkEncoder

    _patch_imwatermark()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        fail(3, f"OpenCV kann {src} nicht oeffnen.")
    enc = WatermarkEncoder()
    enc.set_watermark("bytes", payload)

    expected_bytes = w * h * 3
    stderr_log = Path(str(dst) + ".stderr")
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps,
        "-i", "-",
        "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
        "-bitexact",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]
    n = 0
    with open(stderr_log, "wb") as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape != (h, w, 3) or frame.dtype.name != "uint8" or frame.nbytes != expected_bytes:
                    proc.kill()
                    fail(3, f"Frame {n}: unerwartete Form {frame.shape}/{frame.dtype} "
                            f"(Aufloesungswechsel im Stream?).")
                marked = enc.encode(frame, "dwtDct")
                try:
                    proc.stdin.write(marked.tobytes())
                except (BrokenPipeError, OSError):
                    break  # ffmpeg vorzeitig beendet -- echter Fehler kommt aus stderr_log
                n += 1
        finally:
            cap.release()
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.wait()
    err_tail = stderr_log.read_bytes().decode("utf-8", "replace")[-800:]
    stderr_log.unlink(missing_ok=True)
    if proc.returncode != 0:
        fail(3, f"ffmpeg-Encode fehlgeschlagen (Exit {proc.returncode}):\n{err_tail}")
    if n != nframes:
        fail(3, f"Frame-Zaehler-Mismatch: {n} verarbeitet, Container meldet {nframes} "
                "(vorzeitiges Decoder-Ende?).")
    print(f"  Watermark: {n}/{nframes} Frames verarbeitet, neu encodiert (crf {crf}, {w}x{h}@{fps})")


def clean_remux(ffmpeg, src, dst):
    run([ffmpeg, "-y", "-v", "error", "-i", str(src),
         "-map", "0:v:0",
         "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
         "-bitexact", "-c", "copy", "-movflags", "+faststart", str(dst)],
        what="Clean-Remux")
    print("  Remux: nur Videostream uebernommen, Metadaten entfernt (verlustfrei, kein Watermark)")


# ----------------------------------------------------------------------- C2PA
def c2pa_sign(signer, src, dst, title):
    from c2pa import Builder

    manifest = {
        "claim_generator_info": [GENERATOR],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": IPTC_TRAINED,
                            "softwareAgent": GENERATOR,
                        }
                    ]
                },
                # Pflicht: als VOM ERSTELLER erklaerte Assertion einbetten.
                # Ohne dieses Flag landet sie in gathered_assertions
                # ("uebernommen") -- empirisch verifiziert 2026-08-04.
                "created": True,
            }
        ],
    }
    try:
        with Builder.from_json(json.dumps(manifest)) as builder:
            builder.sign_file(src, dst, signer)
    except Exception as e:
        fail(4, f"C2PA-Signatur fehlgeschlagen: {e}")
    print("  C2PA: Manifest eingebettet (c2pa.created, trainedAlgorithmicMedia, created-Assertion)")


# --------------------------------------------------------------------- Verify
def verify(ffprobe, path, payload, watermarked, leaf_serial):
    problems = []
    from c2pa import Reader

    try:
        with Reader(path) as reader:
            state = reader.get_validation_state()
            state_name = getattr(state, "name", None) or (str(state) if state is not None else None)
            if state_name not in ("Valid", "Trusted"):
                problems.append(f"validation_state={state_name!r} (erwartet Valid/Trusted)")
            if not reader.is_embedded():
                problems.append("Manifest nicht eingebettet")
            data = json.loads(reader.json())
            active_label = data.get("active_manifest")
            manifest = (data.get("manifests") or {}).get(active_label)
            detailed = json.loads(reader.detailed_json())
    except Exception as e:
        return [f"C2PA-Read: {e}"]

    if not manifest:
        problems.append("kein aktives Manifest")
    else:
        gen = (manifest.get("claim_generator_info") or [{}])[0].get("name")
        if gen != GENERATOR["name"]:
            problems.append(f"claim_generator={gen!r}")
        actions = []
        for a in manifest.get("assertions", []):
            if str(a.get("label", "")).startswith("c2pa.actions"):
                actions += a.get("data", {}).get("actions", [])
        if not any(a.get("action") == "c2pa.created" and a.get("digitalSourceType") == IPTC_TRAINED
                   for a in actions):
            problems.append("KI-Kennzeichnung (c2pa.created + trainedAlgorithmicMedia) fehlt im Manifest")
        sig = manifest.get("signature_info") or {}
        if str(sig.get("cert_serial_number")) != leaf_serial:
            problems.append(f"Signer-Pinning: Seriennummer {sig.get('cert_serial_number')!r} "
                            f"!= lokales Leaf {leaf_serial}")
        claim = (detailed.get("manifests") or {}).get(active_label, {}).get("claim", {})
        created_urls = " ".join(a.get("url", "") for a in claim.get("created_assertions", []))
        if "c2pa.actions" not in created_urls:
            problems.append("Actions-Assertion nicht in created_assertions (gathered-Regression)")
    if problems:
        pass
    else:
        print("  Verify C2PA: Valid, Manifest-Policy ok, Signer gepinnt, created-Assertion ok")

    if watermarked:
        import cv2
        from imwatermark import WatermarkDecoder
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        checked = []
        for idx in sorted({0, total // 2, total - 1}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                problems.append(f"Watermark-Check: Frame {idx} nicht lesbar")
                continue
            dec = WatermarkDecoder("bytes", len(payload) * 8)
            got = bytes(dec.decode(frame, "dwtDct"))
            if got != payload:
                problems.append(f"Watermark Frame {idx}: erwartet {payload!r}, gelesen {got!r}")
            else:
                checked.append(idx)
        cap.release()
        if checked and len(checked) == len({0, total // 2, total - 1}):
            print(f"  Verify Watermark: Payload {payload!r} aus Frames {checked} dekodiert")

    info = probe(ffprobe, path)
    if info.get("chapters"):
        problems.append("Metadaten: Kapitel im Export")
    bad = {k: v for k, v in (info.get("format", {}).get("tags") or {}).items()
           if k.lower() not in ALLOWED_FORMAT_TAGS}
    for s in info.get("streams", []):
        for k, v in (s.get("tags") or {}).items():
            if k.lower() not in ALLOWED_STREAM_TAGS or len(str(v)) > MAX_TAG_VALUE_LEN:
                bad[f"stream:{k}"] = v
    if bad:
        problems.append(f"Metadaten-Leak: {bad}")
    elif not problems:
        print("  Verify Metadaten: Format/Stream/Chapter sauber")
    return problems


# ----------------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser(description="BotterDancer Export: Watermark + Metadaten-Strip + C2PA")
    ap.add_argument("input", help="Roh-MP4 aus ComfyUI")
    ap.add_argument("--outdir", default=r"C:\ComfyUI\output\export", help="Zielverzeichnis (lokal)")
    ap.add_argument("--no-watermark", action="store_true",
                    help="nur Remux+C2PA (verlustfrei, ohne unsichtbares Watermark)")
    ap.add_argument("--wm-payload", default="BD26", help="Watermark-Payload (1-8 ASCII-Zeichen)")
    ap.add_argument("--crf", type=int, default=16, help="x264-Qualitaet (10-30)")
    ap.add_argument("--overwrite", action="store_true", help="vorhandene Export-Datei ersetzen")
    args = ap.parse_args()

    try:
        payload = args.wm_payload.encode("ascii")
    except UnicodeEncodeError:
        fail(2, "--wm-payload muss reines ASCII sein.")
    if not 1 <= len(payload) <= 8:
        fail(2, f"--wm-payload: 1-8 Zeichen erlaubt, {len(payload)} uebergeben.")
    if not 10 <= args.crf <= 30:
        fail(2, f"--crf {args.crf} ausserhalb 10-30.")

    src = Path(args.input)
    if not src.exists():
        fail(2, f"Eingabe {src} existiert nicht.")
    if str(args.outdir).startswith("\\\\"):
        fail(2, "--outdir darf kein Netzwerkpfad sein -- Export publiziert erst NACH Verifikation lokal.")
    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        fail(2, "ffmpeg UND ffprobe werden benoetigt (PATH oder Fallback-Verzeichnis).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / (src.stem + "_export.mp4")
    if dst.exists() and not args.overwrite:
        fail(2, f"{dst} existiert bereits -- --overwrite zum Ersetzen.")

    signer, leaf_serial = load_credentials()
    w, h, fps, nframes = enforce_contract(probe(ffprobe, src), src)
    if not args.no_watermark and (w < 256 or h < 256):
        # imwatermark/dwtDct verweigert unter 256x256 (RuntimeError tief im Encode)
        fail(2, f"Eingabevertrag: {w}x{h} zu klein fuers Watermark (min. 256x256) — "
                f"--no-watermark nutzen oder groesser rendern.")
    integrity_check(ffmpeg, src)
    print(f"Export: {src.name} ({w}x{h}@{fps}, {nframes} Frames) -> {dst}")

    with tempfile.TemporaryDirectory(prefix="botter_export_", dir=str(outdir)) as td:
        essence = Path(td) / "essence.mp4"
        staged = Path(td) / "staged.mp4"
        if args.no_watermark:
            clean_remux(ffmpeg, src, essence)
        else:
            watermark_and_reencode(ffmpeg, src, essence, payload, args.crf, w, h, fps, nframes)
        c2pa_sign(signer, essence, staged, dst.name)

        problems = verify(ffprobe, staged, payload, watermarked=not args.no_watermark,
                          leaf_serial=leaf_serial)
        if problems:
            for p in problems:
                print("PROBLEM: " + p)
            fail(5, "Verifikation fehlgeschlagen -- nichts publiziert, Export-Ordner bleibt sauber.")
        os.replace(staged, dst)  # Temp liegt im outdir -> selbes Volume, atomar

    size_mb = dst.stat().st_size / 1e6
    print(f"ERFOLG: {dst} ({size_mb:.1f} MB) -- signiert, geprueft, publiziert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
