# BotterDancer Spike â€” Abschlussbericht Deep-Review (2026-07-10)

## 1. Sofort fixen (vor dem nÃ¤chsten Render)

1. **Referenz-Latent klebt im Output â€” TrimVideoLatent fehlt.** Beide Outputs haben 53 statt 49 Frames (ffprobe `botter_iso_00001.mp4` / `botter_smoke_00001.mp4`), +250 ms Zeitversatz, sichtbarer â€žstehende Kreatur"-Erstframe. Mechanismus code-verifiziert: `nodes_wan.py:1153-1159` (trim_latent += ref-Latent).
   **Fix:** `TrimVideoLatent` zwischen KSampler und VAEDecode, `trim_amount` â† WanAnimateToVideo-Output 3 (`trim_latent`). Node existiert im Install (/object_info bestÃ¤tigt). Ein Frame in Post schneiden reicht NICHT (4 kontaminierte Pixelframes, kausal VAE-verschmiert).
2. **GPU-Hygiene â€” in exakt dieser Reihenfolge** (Verifikation hat die Reihenfolge korrigiert): (a) ComfyUI **ohne `--highvram`, mit `--reserve-vram 4`** neu starten â€” unter `--highvram` ist `offload_device=cuda:0` (`model_management.py:880-883`, `model_patcher.py:1138-1142`), `/free` damit ein VRAM-No-Op; (b) erst dann `POST /free {"unload_models":true,"free_memory":true}`; (c) **Ollama stoppen** (`OLLAMA_KEEP_ALIVE=0` / `ollama stop`) + Comfy Desktop + KB-Pipelines pausieren. Beleg: Run 2 lief 36,2â€“41,0 s/it ab Step 1 nach 32.948-MB-Voll-Reload (Log 622â€“640); live gemessen: llama-server 53,5 GB dediziert, 91 % GPU-Util/502 W bei **leerer** ComfyUI-Queue. Erwartung nach Fix: ~2,5 min statt 13:20.
3. **`frame_load_cap` 49â†’81, `length` 49â†’81** â€” aktuell fehlen die letzten ~2 s (~40 %) des Drivers stumm (ffprobe: Driver 150 Frames/5,005 s @29,97; Pose-Video 49/3,0625 s). Erfordert ohnehin neuen DWPose-Export + filter + ffmpeg fÃ¼r 81 Frames â†’ **dabei Crop unten padden** (512x768 oder ~5 %): Frame 21 hat beide Ankles bei y=729,9 auf 720er-Canvas, 130 Joints insgesamt auÃŸerhalb â€” FÃ¼ÃŸe werden am Rand amputiert, genau dort, wo FuÃŸkontakt-QualitÃ¤t herkommt.
4. **CLIPVisionEncode `crop` â†’ "none".** center-Crop behÃ¤lt bei der 832x1216-Referenz nur das mittlere 832x832-Band â€” je ~190 px oben (Kopf) und unten (FÃ¼ÃŸe) erreichen clip_vision_h nie (`clip_model.py:12-21`, /history bestÃ¤tigt crop:"center"). Ein Widget.
5. **WanAnimateToVideo `width` 480â†’512.** Macht Poseâ†’Generation pixelexakt und eliminiert den 16-px/Seite-Crop (`comfy/utils.py:1036-1047`); 512/16=32, gÃ¼ltig.
6. **filter_pose.py: Torso-Anker-Tracker statt per-Frame argmax(bbox_area).** Live gemessen: Pose moduliert die eigene FlÃ¤che 3,2x (78.680â†’252.624 pxÂ²), Trennmarge top1/top2 min. nur 1,48x (Frame 23), Auswahl-Index churnt 1â†’2â†’1â†’0â†’â€¦ â€” der Flip-Mechanismus ist real, es traf nur diesen Clip nicht. Zudem Intent-Mismatch: Datei heiÃŸt `*_center`, Code wÃ¤hlt LARGEST. **Fix:** Anker = mean(Neck, R/L-Hip); Frame 0 = nÃ¤chster zur Canvas-Mitte; dann argmin-Distanz zu EMA-Anker mit Max-Jump-Gate (~12 % Diagonale); largest-bbox nur nach â‰¥3 Lost-Frames. Drop-in-sicher verifiziert: 0 Abweichungen auf allen 49 Frames.
7. **Kernbefund ist konfundiert â€” Single-Variable-Rerun nÃ¶tig.** /history (f7887ad0 vs. e52e96f3): Der Smoke-Run nutzte â€ža person dancing" + Negativ OHNE â€žhuman", der Iso-Run Creature-Prompt + â€žhuman"-Negativ. â€žMulti-Skelett zerstÃ¶rt IdentitÃ¤t" ist aus diesen zwei Runs nicht ableitbar. **Fix:** ein Rerun 3-Skelett-Pose + Creature-Prompt-Paar, seed 42, bevor die Architektur Zwangs-Isolation festschreibt.

## 2. Bald fixen (vor App-Bau)

- **Confidence-Achse ist zerstÃ¶rt:** alle 2.635 Body-Confs im JSON exakt {0.0, 1.0}; Encoder hardcodet 1.0 (`dwpose/__init__.py:195`), Decoder droppt c<1.0 (`:158`) â€” echtes OpenPose-JSON wÃ¼rde stumm leere Posen liefern. Fix upstream: Raw-Scores vor `encode_poses_as_dict` abgreifen; bis dahin strukturelles Gate (Neck+beide Hips Pflicht, SegmentlÃ¤nge â‰¤1,5x trailing-Median = zugleich ChimÃ¤ren-Detektor).
- **prev-Freeze-Fallback dreifach defekt** (`filter_pose.py:26-30`, Pfad in diesem Clip 100 % dead code): max() auf lauter 0-FlÃ¤chen wÃ¤hlt ein Fragment und vergiftet den Track; Freeze-then-Snap bei echtem Ausfall; `kept`-ZÃ¤hler zÃ¤hlt Frozen-Frames mit (â€žDONE 49/49" lÃ¼gt). Fix: Two-Pass, Gaps â‰¤8 Frames interpolieren, darÃ¼ber laut scheitern, ZÃ¤hler getrennt (fresh/interpolated/failed).
- **sys.path-Hack auf private Node-Internals:** Vendoring der ~150 benÃ¶tigten Zeilen ins Repo (oder Commit-Pin + Import-Assert); SRC/OUTDIR/fps via argparse; meta.json-Sidecar fÃ¼r den ffmpeg-Schritt.
- **OUTDIR-Stale-Tail:** Rerun mit weniger Frames lÃ¤sst alte PNGs stehen, ffmpeg `%04d` konkateniert sie stumm ins Video. Fix: OUTDIR vor Schreiben leeren; Skript besitzt den ffmpeg-Schritt mit explizitem Frame-Count.
- **Pose-Conditioning verlustfrei:** h264 yuv420p halbiert Chroma auf 1â€“5-px-Linien; Stroke-Dicke inkonsistent (Inline-Pfad 720x1012 vs. Filter 512x720, fixe stickwidth 4 in `util.py:103,147`). Fix: PNG-Sequenz (VHS Load Images) oder ffv1/crf-0, gerendert in ZielauflÃ¶sung.
- **Reproduzierbarkeit:** yolox_l/dw-ll onnx wurden ungepinnt von HF nachgeladen (Log 144â€“173), ORT-CUDA nicht bit-deterministisch, BinÃ¤r-Quantisierung flippt Keypoints an der Schwelle. Fix: Modelle per SHA256 pinnen, vollstÃ¤ndiger Cache-Key (Source-Hash, Crop, VHS-Params, Modell-Hashes, Filterversion, Canvas).
- **Admission nur auf physisches VRAM bauen:** `/system_stats` meldete 64,2 GB frei bei physisch 16,1 GB (48-GB-WDDM-Fiktion, `model_management.py:1579`). Und: Contention ist auch COMPUTE (91 % Util bei leerer Queue) â€” Gate braucht nvidia-smi/Perf-Counter + Utilization, exklusive Render-Fenster.
- **`save_metadata=False` bzw. Clean-Remux vor Export:** beide MP4s tragen Prompt, Modellnamen und kompletten Graph als Metadaten (VHS-Default-Warnung im Log). Umkehrproblem zu bekanntem Bug 6, eigener Fix.
- **Log-Capture-Falle:** comfyui.log ist UTF-16LE (PS-5.1-`>>`) â€” grep findet darauf NICHTS; Eviction-Marker fÃ¼rs Monitoring ist â€žN models unloaded." (INFO, `model_management.py:785`), nicht â€žUnloading" (DEBUG). Fix: `-Encoding utf8` loggen, INFO-Marker keyen.
- **steps 15â†’20 fÃ¼r Quality-PÃ¤sse** (offizielle Wan2.2-Templates: 20; verifiziert auf Disk); Draft-Speed Ã¼ber Distill-LoRA (cfg 1.0/4 Steps), nicht Ã¼ber weniger Base-Steps. Kein Distill-LoRA auf Disk (verifiziert, nur Flux-Turbo).
- **Benchmarks neu ziehen:** alle Zahlen beschreiben 3,06 s statt 5 s (~1,6x zu optimistisch) und sind durch Co-Tenants kontaminiert â€” bei sauberer GPU auf 81 Frames re-benchen, s/it pro Run loggen.

## 3. Bekannte Bugs, jetzt live belegt

- **Bug 2 (keine globale VRAM-Admission / Zwei-Queue-Design):** jetzt hart belegt und verschÃ¤rft â€” gemessene Dritt-Tenants auÃŸerhalb jeder Queue (llama-server 53,5 GB, zweite ComfyUI-Instanz, Docker), `/system_stats` als Datenquelle unbrauchbar (48-GB-LÃ¼ge), und Contention frisst auch Compute (91 %/502 W bei leerer Queue).
- **Foot-Skate-Bug:** neuer Input-seitiger Beleg (angrenzend, nicht identisch): in 5/49 Frames verlassen die Ankles des Gewinners den Canvas (Frame 21: y=729,9/720) â€” das Modell muss Bodenkontakt halluzinieren, weil das Pose-Video ihn nie sah.
- **Bug 6 (C2PA stirbt im Re-Encode):** kein neuer Beleg, aber der Spiegelbefund (save_metadata-Leak, Abschnitt 2) gehÃ¶rt daneben ins selbe Kapitel â€žMetadaten beim Export".
- FÃ¼r die Ã¼brigen bekannten Bugs lieferte dieser Pass keine neuen Belege.

## 4. Von der Verifikation gekippt

- **â€ž~6 % horizontaler Squash 512â†’480":** gekippt â€” es ist ein 16-px/Seite-CROP, kein Squash (`utils.py:1036-1047`), und fÃ¼r diesen Clip messbar harmlos (0/2938 Keypoints in den Crop-BÃ¤ndern). Downgrade auf latentes Config-Risiko.
- **â€žFlux-Residenz verursachte die 5x-Verlangsamung" (Briefing-Narrativ):** gekippt â€” Log beweist Flux lief VOR dem schnellen Smoke-Run (Zeilen 228â€“260 vor 261).
- **Beleg â€žgrep fand null Unloading-Zeilen":** doppelt nichtig (UTF-16LE-Log + DEBUG-Level-Message) â€” Schluss Ã¼berlebte nur auf dem korrekten INFO-Marker (â€žN models unloaded.": 0 Vorkommen).
- **â€ž~100+ GB committed bei Run-2-Start":** von Messung auf Inferenz herabgestuft (aktuell nur 33,9 GB committed beweist spÃ¤tere Freigabe); Oversubscription-Arithmetik hÃ¤lt auch konservativ.
- **UrsprÃ¼ngliche Fix-Reihenfolge (/free vor Relaunch):** gekippt â€” unter `--highvram` ist /free ein VRAM-No-Op; Relaunch zuerst.
- **gemma2-Residenz WÃ„HREND Run 2:** unbewiesen (Server-Start 13:13:13, 46 s NACH Run-2-Ende) â€” bleibt Inferenz, ehrlich als solche markiert.
- **fps-Drift-Sorge:** entkrÃ¤ftet â€” Akkumulator-Resampling hat null kumulativen Drift, â‰¤33 ms Jitter; 16 fps ist korrekt fÃ¼r A14B/Animate.
- **Identity-Flip in DIESEM Clip:** fand nicht statt (0 Abweichungen) â€” der Mechanismus ist gemessen, der Flip selbst Inferenz fÃ¼r andere Clips.

## 5. Neue Features (dedupliziert, gerankt)

1. **[L] Full-Crew Casting** â€” drei Einzel-Pose-Tracks + drei Referenzen + character_mask/background_video (Nodes verifiziert vorhanden): die Dilution-SchwÃ¤che wird zur Formation aus drei Kreaturen â€” das teilbarste Output, das die App haben kann.
2. **[M] Dancer Picker + Full-Frame-Detect** â€” DWPose einmal aufs Vollbild, Track-IDs, Klick wÃ¤hlt den Dancer und seedet den Tracker; Crop-Ã„nderungen werden gratis; die natÃ¼rliche UI fÃ¼r K-Pop-Formationen.
3. **[M] Render Forecaster + Ein-Klick-/free** â€” ehrliche ETA aus nvidia-smi-Livezustand (heute kalibrierbar: ~8â€“10 vs. ~40 s/it); fÃ¼r Solo-User ist Wahrheit + ein Klick besser als Scheduling.
4. **[M] Auto-QA Gate** â€” First-Frame-Leak-SSIM, Identity-Hold via clip_vision, Pose-Track-Sanity vor dem Render: beide abgedeckten Fehlerklassen sind heute live aufgetreten.
5. **[M] Provenance-/Timing-Manifest pro Job** â€” ein Schema ist zugleich Cache-Key, Beat-Grid-Tabelle und â€žexakt diesen Clip regenerieren"; ohne ist jeder Clip ein Waisenkind.
6. **[M] Long-Take Chaining** â€” continue_motion + video_frame_offset (Schema verifiziert) fÃ¼r Takes beliebiger LÃ¤nge; â€žTake deckt 0:00â€“0:05 (2 Chunks)" statt stummem Abschneiden.
7. **[M] Audio-/Beat-Pipeline** â€” Beats auf dem VOLLEN Original-Audio, via Manifest-Frame-Map gemappt, nach Referenz-Trim gemuxt; Beat-Sync ist ohne das unbaubar.
8. **[M] Proportion Pre-Warp** â€” Limb-Segmente vor dem Zeichnen auf Kreaturen-Anatomie skalieren; greift die beobachtete Humanisierung geometrisch an der Wurzel an.
9. **[M] Face Channel Toggle (OFF/DRIVE/PROTECT)** â€” der ungenutzte face_video-Input macht das Gesicht zum expliziten Regler statt Kollateralschaden.
10. **[M] Room-Keeper** â€” background_video mit gemattetem Original-Raum (Matting-Nodes verifiziert installiert): Kreatur tanzt im echten Practice-Room, groÃŸer Perceived-Quality-Sprung ohne ModellÃ¤nderung.
11. **[M] lightx2v-Draft-Lane** â€” Distill-LoRA (Download nÃ¶tig, keins auf Disk), 4 Steps/cfg 1.0: ~60â€“90 s pro Iteration statt 2,5â€“13 min.
12. **[S] Prompt Autopilot** â€” lokales VLM (qwen2.5-vl auf Disk verifiziert) schreibt Prompt aus der Referenz; Prompt-Wortlaut ist nachweislich load-bearing (Smoke-Humanisierung).
13. **[S] Draft Stamp** â€” Take-ID/Seed/Steps im Pixelbereich auf Drafts; Ã¼berlebt per Konstruktion jeden Re-Encode, Finals rendern ohne den einen Node.

*Hinweis: Tracker, Gap-Interpolation, QC-Sidecar, Trim-Fix, GPU-Janitor, Launch-Flags und One-Graph-Fix-Pass aus dem Pool sind Bugfixes (Abschnitte 1â€“2), keine Features â€” dort einsortiert statt doppelt gelistet.*
