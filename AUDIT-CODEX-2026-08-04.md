# Externes Codex-Audit der C2PA-Export-Tools (2026-08-04)

Setup: Codex CLI 0.144.0-alpha.4 (`gpt-5.6-sol` @ ultra), read-only Sandbox auf
isolierter Kopie von `export_clip.py` + `make_c2pa_cert.py` + CONTEXT.md
(ohne `assets\keys\`). Parallel eigener Re-Attack-Pass (Fable); 4 Findings
deckungsgleich, Rest komplementär. Alle Überraschungs-Findings wurden vor dem
Fix **empirisch am System verifiziert**, nicht blind übernommen.
Ergebnis: v2 beider Tools, Happy-Path + 4 Negativ-Tests grün.

## HOCH — Status

| # | Finding (Kurzform) | Verifikation | Status |
|---|---|---|---|
| 1 | KI-Kennzeichnung landete in `gathered_assertions` statt `created_assertions` | detailed_json des echten Exports | **GEFIXT**: `"created": true` in der Assertion; Verify-Gate prüft Platzierung als Regression |
| 2 | Verify-Gate authentifiziert Signer/Policy nicht; `str(None)` passiert Substring-Check | Code-Read | **GEFIXT** (pragmatisch): State-Whitelist Valid/Trusted mit None-Guard, `is_embedded`, Manifest-Policy (Action+DST+Generator), Seriennummern-Pinning gegen lokales Leaf. OFFEN: echte `trust_anchors`-Config (Trusted erzwingen) |
| 3 | Metadaten-Strip unvollständig (nur global), Prüfung fail-open | ffmpeg-Doku + Code | **GEFIXT**: `-map_metadata:s -1`, `-map_chapters -1`, `-bitexact`, explizites `-map 0:v:0`; ffprobe Pflicht; Sweep über Format+Stream+Chapter mit Whitelist und 64-Zeichen-Limit. OFFEN: SEI-Tiefeninspektion |
| 4 | Publikation vor Verifikation; Kollisionen; UNC-Ziele | Code-Read | **GEFIXT**: Staging im Temp (im outdir → selbes Volume), `os.replace` erst nach bestandener Prüfung, Kollisions-Guard (`--overwrite`), UNC-Reject |
| 5 | Vorzeitiges Decoder-EOF wird als Erfolg gewertet | Code-Read | **GEFIXT**: `-xerror -err_detect explode`-Volldecode beider Pfade + hartes `n == nb_frames` |
| 6 | dwtDct-Upstream vertauscht Wavelet-Bänder | **Messung**: auf unserem Content folgenlos (PSNR 40,3 dB identisch mit/ohne Fix — nur U-Chroma betroffen, Pastell hat dort kaum Detail); Quellcode bestätigt Vertausch | **GEFIXT** (Monkeypatch + Clipping) — Schwere für unseren Content niedriger als von Codex angesetzt, relevant erst für Room-Keeper-Content |
| 7 | Audio/Nebenstreams/fehlender Videostream ungeprüft | Negativ-Test T3 | **GEFIXT**: Eingabevertrag „genau 1 Videostream", Audio → harter Fehler mit Hinweis |
| 8 | Watermark nur an Frame 0 geprüft | Code-Read | **GEFIXT**: Frames 0/Mitte/letzter, alle müssen dekodieren |
| 9 | Privater Key an geerbter DACL | Code-Read | **GEFIXT**: `icacls /inheritance:r`, nur User+SYSTEM. OFFEN: DPAPI/CNG statt Klartext-PKCS8 |
| 10 | Cert-Tool-„Idempotenz" destruktiv bei Teilzuständen | Code-Read | **GEFIXT**: echte Krypto-Konsistenzprüfung (Key↔Leaf↔CA↔Chain), Teilzustand = harter Abbruch ohne `--force`, atomare Writes, Selbsttest |

## MITTEL/NIEDRIG — Status

- VFR-Input, 10-bit/HDR/Rotation: **fail-closed** über Eingabevertrag (CFR, 8-bit 4:2:0, keine Rotation) statt stiller Normalisierung.
- Auflösungswechsel im Stream: per-Frame Shape/dtype/nbytes-Check.
- BrokenPipe/stderr-Deadlock: stderr in Datei, BrokenPipe sauber behandelt, Fehler aus Log.
- BMFF-Hash ≠ Vollfile-Integrität: Kommentar korrigiert. OFFEN: strukturelle Box-Prüfung bzw. extern gepinnter SHA-256, falls je nötig.
- Credential-Preflight: Schlüssel werden vor jeder Pixelarbeit geladen/validiert.
- Payload/CRF/Argument-Validierung: vorab erzwungen (1–8 ASCII, CRF 10–30).
- Root im x5chain: **OFFEN/AKZEPTIERT** (Kette bleibt eingebettet; nur-Leaf wäre spez-konformer, Nutzen privat gering).

## Testabdeckung v2 (2026-08-04)

Happy-Path Watermark (cam180: 81/81, Verify 3 Frames, Policy+Pinning ok),
Happy-Path `--no-watermark` (cam0-draft), Negativ: Kollision ohne `--overwrite`,
Payload >8 Zeichen, Input mit Audiospur, UNC-`--outdir` — alle Exit 2 mit klarer Meldung.

## Lehren

- `codex exec` mit Desktop-App-Config: Standalone-CLI 0.136 kennt `ultra`/`max` nicht —
  die App-eigene CLI unter `%LOCALAPPDATA%\OpenAI\Codex\bin\<hash>\codex.exe` nutzen.
- Externe Findings mit drastischen Zahlen erst am eigenen Content nachmessen
  (dwtDct-Bug: real, aber für uns 40 dB PSNR = unsichtbar).
- `Reader.get_validation_state()` kann `None` liefern; Enum-Namen über `.name` ziehen.
