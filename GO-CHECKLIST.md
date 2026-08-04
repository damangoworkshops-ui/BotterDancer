# GO-Checkliste — sobald die GPU frei ist [ERLEDIGT 2026-07-10, Doku nachgezogen 2026-07-19]

> **Status 2026-07-19:** Schritte 0–5b sind laut Datei-Evidenz (Audit 18./19.07.) am 10.07.
> vollständig durchgelaufen — nur die Checkboxen wurden nie angekreuzt; jetzt nachgeholt.
> Offen ist allein 5c (Benchmark in RESEARCH.md). Für NEUE Läufe gilt der Weg aus
> `FIXES-2026-07-19.md`: `tools\submit_workflow.py --preset …` statt manuellem POST.
> Vorbereitet 2026-07-10 (Trockenphase). Assets: `botter_pose_all.mp4` (Dilution),
> `botter_pose_center_v2\`, `creature_ref.png`, Workflows in `workflows\`.

## 0. GPU freiräumen (einmalig)
- [x] Anderen Job abwarten / bestätigen, dass er fertig ist.
- [x] **Ollama/llama-server entladen**: `ollama stop` bzw. `OLLAMA_KEEP_ALIVE=0` (belegte 53,5 GB + 91 % Util!).
- [x] Alte ComfyUI-Instanz beenden, neu starten mit **`tools\start_comfy.ps1`** (ohne `--highvram`, mit `--reserve-vram 4`, UTF-8-Log).
- [x] Check: `nvidia-smi` → VRAM-Nutzung nahe 0, keine fremden Prozesse.

## 1. Dilution-Rerun (klärt den konfundierten Kernbefund, ~4 min)
- [x] `workflows\archive\rerun_dilution.json` senden (2026-07-19 archiviert; Experiment ist gelaufen).
- [x] Vergleich gegen `botter_iso_00001.mp4` (gleicher Prompt, Seed, Steps — einzige Variable: 3 Skelette statt 1).
- [x] **Ergebnis (10.07.):** Prompt steuert die Klasse, Identität verteilt sich lose → Isolation bleibt Pflicht für exakte Identität; Multi-Skelett = Full-Crew-Pfad.

## 2. Pose-Neuextraktion volle Länge (~1–2 min)
- [x] `workflows\save_pose_81.json` senden → erzeugt `output\botter_pose81_*.json` (~80 Frames).

## 3. Filter v2 auf 81 Frames (CPU, Sekunden)
```powershell
C:\ComfyUI\venv\Scripts\python.exe C:\Users\chris\Documents\BotterDancer\tools\filter_pose_v2.py `
  --src (Get-ChildItem C:\ComfyUI\output\botter_pose81_*.json | Sort LastWriteTime -Desc | Select -First 1).FullName `
  --outdir C:\ComfyUI\input\botter_pose_center_v2
```
- [x] Zähler prüfen: `track`-Anteil hoch, `lost/interpolated` niedrig, Abbruch = Trackerproblem. (79× track, 0 lost)

## 4. Final-Render mit allen 6 Fixes (~2,5 min erwartet, vorher ~13)
- [x] `workflows\wan_animate_final_v2.json` senden.
- [x] **Erfolgskriterien:** exakt 81 Frames im Output (ffprobe-belegt); Füße sichtbar; Identität hält.
- [x] s/it notiert: **15,7 s/it @512×768×81/20 Steps** (= normalisiert ~8,4 auf alte 480p-Basis — Erwartung „8–10“ galt für den kleineren Workload).

## 5. Danach (Kür)
- [x] lightx2v-Distill-LoRA: liegt auf Disk und ist in `wan_animate_draft.json` verdrahtet — Draft-Lane validiert (~41 s Gesamt).
- [x] RIFE-Interpolation 16→32 fps auf dem Final getestet (`botter_final_32fps_00001.mp4`).
- [ ] Benchmarks für RESEARCH.md aktualisieren (81 Frames, saubere GPU) — **einziger offener Punkt**.

## v3-Kamera-Abnahme [ERLEDIGT 2026-08-03]
- [x] Alle 5 Winkel (0/60/90/120/180) via `submit_workflow.py --preset camN-draft` gerendert: je 81 Frames @16 fps = 5,06 s Echtzeit — Zeitlupen-/Abschneide-Bug weg.
- [x] Seitenrichtigkeit belegt: Skelett Frame 51 (t=3,2 s) zeigt grünes (rechtes) Bein links im Bild, blaues (linkes) rechts — korrekt für Frontalansicht; Render folgt (Arme-kreuzen 2,5–3,2 s, Arme-hoch 4,0–4,5 s synchron zum Treiber).
- [x] **NEUER BEFUND — Draft-Lane versagt bei Rückwinkeln:** cam120-Draft (4 Steps) kollabiert in sitzende Haustier-Pose; Full-Quality (20 Steps, `botter_cam120_00002.mp4`) folgt demselben Skelett einwandfrei. Ursache-Kandidat: kopfloses Skelett (Face-Hysterese unterdrückt bei Rückansicht alle Kopfpunkte) = schwaches Signal, das 4 Steps nicht mehr auflösen. **Regel: Winkel ≥120° immer Full-Quality, Draft nur für 0–90°.**
- [x] Alte Pose-Sets (gespiegelt/150f) nach Abnahme verschoben nach `C:\ComfyUI\input\_stale_pose_sets_pre_v3\` (reversibel).
- [x] cam180-Weichheit GELÖST (03.08. nachmittags): Flux-Rück-Referenz generiert (`workflows\flux_ref_back.json`, 3 Prompt-Runden, Sieger = Kandidat 00010 → `C:\ComfyUI\input\creature_ref_back_v2.png`), beide cam180-Presets umgehängt, Full-Render `botter_cam180_00002.mp4`: 81/81 Frames Rückansicht (9×9-Mosaik geprüft), kein Gesichts-Flackern mehr, scharf. Rest-Artefakt (kosmetisch): Horn-Riffeltextur klebt in den ersten ~2 s an den Armen. Achtung: ComfyUI-Desktop-App (v0.30, eigene Pfade) belegte nach Reboot Port 8188 — Projekt-Instanz lief deshalb auf 8190 (`submit_workflow.py --server http://127.0.0.1:8190`, Schema `http://` ist Pflicht).
- [ ] Figur wirkt durch FOV-sichere Distanz klein im Frame — bei Bedarf Sets mit `--distance-scale 0.7` neu generieren.

## C2PA/Watermark-Export [ERLEDIGT 2026-08-04, v2 nach Doppel-Audit]
> **v2 (Abend):** Externes Codex-Audit (10 HOCH-Findings) + eigener Re-Attack gemerged →
> beide Tools gehärtet: Eingabevertrag fail-closed, Verify-Gate mit Signer-Pinning +
> created-Assertion-Check, Staging-then-Publish (Export-Ordner enthält nie Ungeprüftes),
> dwtDct-Upstream-Fix, Key-DACL, Cert-Konsistenzprüfung. Details + offene Restpunkte:
> `AUDIT-CODEX-2026-08-04.md`. Alte Exporte vor v2 ggf. neu exportieren (Assertion lag
> in gathered statt created).
- [x] `tools\make_c2pa_cert.py`: lokale CA + ES256-Leaf (EKU emailProtection) → `assets\keys\` (gültig bis 2036; `c2pa_leaf.key` nicht weitergeben).
- [x] `tools\export_clip.py`: terminaler Export-Schritt (Review-Fix #6) — unsichtbares dwtDct-Watermark (Payload `BD26`, alle Frames) + Re-Encode crf 16 + Metadaten-Strip in einem Pass, DANACH C2PA-Signatur (c2pa.created, trainedAlgorithmicMedia), danach Auto-Verify (Manifest Valid + Watermark-Rückleseprobe + Metadaten-Leak-Check). `--no-watermark` = verlustfreier Remux+Signatur.
- [x] Getestet: cam180-Export (Watermark dekodierbar aus Frame 0 UND 40, visuell unsichtbar), cam0-Draft ohne Watermark. Exporte → `C:\ComfyUI\output\export\`.
- Stack: c2pa-python 0.37.2 + invisible-watermark 0.2.0 + PyWavelets (alle `--no-deps` ins ComfyUI-venv, torch unangetastet). Stolperfalle: `ta_url` muss per Direktzuweisung `None` (NULL) sein — Leerstring → "Signature: empty string".
- Einordnung: Art. 50(2)-Pflicht (maschinenlesbare KI-Kennzeichnung) damit erfüllbar; lokale Vertrauenskette zeigt bei fremden Verifiern "unbekannter Aussteller" — für Privatnutzung unerheblich. **Regel: Kein Clip verlässt die Maschine außer aus `output\export\`.**

## Voll-Audit + Fix-Runde [ERLEDIGT 2026-08-04 abends]
- [x] Codex-Voll-Audit (Masterprompt): 11 Findings, per 11-Agenten-Workflow alle CONFIRMED, alle gefixt + 13 Tests grün. Details: `AUDIT-CODEX-VOLL-2026-08-04.md`.
- **Wichtigste Verhaltensänderungen:** (1) Alle Tools finden die Projekt-Instanz jetzt per Fingerprint selbst (`tools/comfy_target.py`) — Desktop-App auf 8188 wird nie mehr angesprochen; `start_comfy.ps1` weicht bei Fremdbelegung auf 8190 aus. (2) Unresampelte Pose-Ordner und RIFE-fps-Mismatch sind jetzt harte Preflight-FEHLER. (3) Draft-Presets ≥120° gesperrt. (4) Pose-Regenerierung ist transaktional (Lock + Temp + Swap) mit Synthetik- und Sichtbarkeits-Gates. (5) SavePoseKps-Läufe verifizieren ihre Output-JSON selbst.
- Offen/bewusst: workloadabhängige Dashboard-Baselines, genereller Artefaktvertrag, Tests ins Projekt einchecken, git-Repo-Frage.

## Offene Erkenntnisse aus dem Review (nicht vergessen)
- `/system_stats` lügt unter WDDM — VRAM-Checks immer via `nvidia-smi`.
- Altes `comfyui.log` (Scratchpad) ist UTF-16LE — mit PowerShell lesen, nicht grep.
- VHS bettet Prompt+Graph in MP4-Metadaten → erledigt `tools\export_clip.py` jetzt automatisch (manuell: `ffmpeg -i in.mp4 -map_metadata -1 -c copy out.mp4`).
- Confidence-Werte im Pose-JSON sind binär (Encoder hardcodet 1.0) — echter Confidence-Abgriff = Upstream-Arbeit.
