# BotterDancer — Technische Recherche & Konzept

> Stand: 2026-07-04. Quelle: Multi-Agent-Web-Recherche (11 Agenten, primärquellen-verifiziert).
> Dieses Dokument fasst den Tech-Stack, die Lizenz-Landkarte und die getroffenen
> Entscheidungen zusammen. Vollständige Agenten-Rohberichte siehe Git-History dieser Datei
> bzw. Session-Scratchpad.

## Was ist BotterDancer?

Lokale Windows-App: Tanzvideo (YouTube-Link oder Datei) rein → Bewegungen der Tänzer
analysieren → neue KI-Tänzer generieren (random oder mixbare Templates), konditioniert auf
ein Startbild → 5–10-s-Clips mit einstellbarer Dauer, Kamerawinkel und Konsistenz-Kontrolle.
Highlight: **Choreographer** — Tool zum Editieren/Neu-Komponieren der Tanzbewegungen (Regie führen).

**Zielhardware (bestätigt 2026-07-04): RTX PRO 6000 Blackwell Workstation, 96 GB VRAM,
Treiber 595.97.** Das ist Workstation-Klasse, nicht Consumer 12–32 GB — die
VRAM-/Quantisierungs-Tabellen unten stammen aus der ursprünglichen Recherche und gelten als
Untergrenze für Portabilität; auf dieser Karte läuft **alles unquantisiert (fp16/bf16)**.
Siehe Abschnitt „Konsequenzen der Hardware". **Nutzung privat** (kein kommerzieller Vertrieb
geplant → Non-Commercial-Lizenzen sind aktuell kein Blocker).

### Konsequenzen der 96-GB-Hardware

- **Keine Quantisierung nötig:** Wan 2.2 14B / SCAIL-2 / Hunyuan 1.5 / LTX-2 alle in voller
  Präzision, 720p+ komfortabel. GGUF/fp8/Block-Swap-Logik entfällt (nur relevant, falls die
  App je auf schwächerer Hardware laufen soll).
- **Schwere Tools jetzt in Reichweite:** MAGREF (~70 GB, Multi-Subjekt-Referenz),
  Uni3C (~50 GB, 3D-konsistente Kamera+Hintergrund), TC-Light full (300 Frames @720p ~40 GB),
  TrajectoryCrafter (28 GB), FlashVSR im vollen Modus. Diese waren für Consumer-Karten
  ausgeschlossen.
- **Parallele Pipeline-Stufen:** Extraktion + Generierung müssen nicht mehr sequenziell laufen;
  mehrere Modelle gleichzeitig im Speicher möglich.
- **LoRA-Training trivial** (Charakter-Templates); große Batches.
- **CAVEAT Blackwell (sm_120):** braucht CUDA 12.8+ und PyTorch 2.7+ (bzw. Nightly). Manche
  ComfyUI-Custom-Nodes mit vorkompilierten CUDA-Kerneln (Block-Sparse-Attention à la FlashVSR,
  SageAttention) müssen evtl. für sm_120 neu gebaut werden. Treiber 595.97 ist aktuell genug.
- **Design-Empfehlung:** App-intern trotzdem eine VRAM-Stufen-Abstraktion behalten, falls das
  Projekt je portabel werden soll — aber Default-Profil = „Ultra/unquantisiert".

---

## Getroffene Entscheidungen (2026-07-04)

1. **Video-Engine:** Drei umschaltbare Backends — **Wan 2.2** (Animate/VACE, einzige mit
   nativem Pose-Transfer, Choreographer-Pfad läuft immer hierüber), **HunyuanVideo 1.5**
   (I2V, kein Pose-Modus), **LTX-2** (schnell, bis 20 s, 4K, synchrones Audio). UI mit
   Capability-Flags pro Engine; alle drei via ComfyUI; WanGP als Referenz-Implementierung.
2. **Kennzeichnung:** Unsichtbares Watermark + C2PA fest im Export (EU-AI-Act-Art.-50(2)-Pflicht).
   Sichtbares Watermark manuell/optional nachträglich.
3. **Audio:** Export bekommt immer eigenes/rechtefreies Audiofile untergelegt; Original-Song
   wird nie exportiert (GEMA-Problem entfällt). Original-Audio nur lokal für Beat-Analyse.
4. **Architektur:** ComfyUI headless als Engine (per API, GPL-Isolation), eigene
   FastAPI-Orchestrierung, Tauri/Electron + React + three.js-Frontend.

---

## 1. NVIDIA-Screenshot — Realitätscheck

Alle drei Modelle sind **echt** (2026er Releases), aber zwei sind das falsche Werkzeug.

| Behauptung | Realität | Urteil |
|---|---|---|
| **LocateAnything-3B** | Existiert (Mai 2026). Objekterkennung/Grounding, keine Pose/Motion/Generierung. "12,7 Boxen/s" auf H100 gemessen. Lizenz: **non-commercial**. | Falsches Tool |
| **Cosmos 3** | Existiert (Juni 2026). Weltmodell für Robotik/Physical AI. Kleinste Variante (Nano 16B) zielt auf 96-GB-Workstations. OpenMDW-1.1 (permissiv). | Falsches Tool |
| **Kimodo** (nv-tlabs) | Existiert (März 2026). 3D-Motion-Diffusion aus Text+Keyframes+Wegpunkten, mit **Timeline-Editor**. Apache-2.0-Code, SOMA-Weights kommerziell OK, ~17 GB VRAM. **Korrektur: KEIN Video-Input** ("2D" = Bodenwegpunkte), gibt Skelett-Motion aus (kein Video). | Baustein für Choreographer |
| **TAO / DeepStream** | Real, aber Video-Analytics (Retail/Inspektion). | Streichen |

Der Screenshot verschweigt das eigentliche Kern-Tool: **Wan2.2-Animate** (Charakter-Bild +
Tanzvideo → neuer Tänzer).

---

## 2. Empfohlene Pipeline (alles lokal)

```
Ingest (Datei-Upload primär; Szenenschnitt PySceneDetect+TransNetV2; Tänzer-Wahl SAM 2)
  → Motion-Extraktion (RTMW/DWPose 2D + GVHMR 3D → Rotationen, Root-Traj., Fußkontakt)
  → Choreographer (Beat-Timeline, schneiden/loopen/spiegeln/retimen/blenden, 3D-Kamera)
  → Skelett aus gewählter Kamera rendern → Wan-Animate/VACE (Startbild + Charakter-Ref)
  → Post (SeedVR2-Upscale 1080p → RIFE-Interpolation → C2PA-Signierung → Export)
```

**Zentrale Designentscheidung: Motion in 3D halten.** Reine 2D-Pipelines nageln dich an die
Kamera des Quellvideos. In 3D fallen Kamerawinkel, Move-Remixing und Retargeting aus einer
Datenstruktur.

---

## 3. Video-Generierung (Kern)

| Modell | Pose-Transfer? | Lizenz | VRAM (quant.) | Rolle |
|---|---|---|---|---|
| **Wan2.2-Animate-14B** | Ja, nativ | Apache-2.0 | 8–24 GB | **Default-Engine** |
| **Wan2.2-VACE-Fun-A14B** | Ja (Pose-Control-Video) | Apache-2.0 | 8–24 GB | LoRA-/Konsistenz-Pfad |
| **SCAIL-2** (Zhipu, Jun 2026) | Ja (3D-Pose) | Apache-2.0 | 14B-Klasse | Experimentell: Spins, stilisiert, Multi-Tänzer |
| **HunyuanVideo 1.5** | Nein | Tencent Community (EU ausgeschlossen*) | fp8 ~9–14 GB | I2V-Engine |
| **LTX-2** | Unbelegt | LTX-2 Community (<10M ARR frei) | ~16 GB | Schnell, lang, Audio |

\* EU-Ausschluss betrifft kommerziellen Vertrieb — für private lokale Nutzung irrelevant.

**Nicht genommen:** MimicMotion/StableAnimator/MusePose/Champ (SVD-Ära, überholt),
CogVideoX/Mochi (kein Pose-Pfad).

**Ökosystem:** ComfyUI + Kijai WanVideoWrapper (Bleeding Edge: Block-Swap, Context-Windows,
fp8) oder WanGP (bestes Low-VRAM-Standalone ab 6 GB).

---

## 4. Motion-Extraktion

| Tool | Lizenz | Rolle |
|---|---|---|
| **RTMW** (mmpose) | Apache-2.0 | 2D-Ganzkörper (133 kpts), Windows-freundlich |
| **DWPose** | Apache-2.0 | 2D-Standard für ControlNet/Wan-Preprocessing |
| **GVHMR** (zju3dv) | **ZJU non-commercial** | Bester 3D-Extraktor (weltverankert, SMPL) |
| **SAM 3D Body + MHR** (Meta) | SAM-Lizenz / **MHR Apache-2.0** | Kommerziell sauberer 3D-Pfad |
| **WHAM / TRAM** | MIT-Code (+ SMPL) | Fußkontakt / metrische Trajektorie |
| **SAM 2 / SAMURAI** | Apache-2.0 | Tänzer-Auswahl per Klick, Tracking |
| **PySceneDetect + TransNetV2** | BSD-3 / MIT | Szenenschnitt-Erkennung |

**Lizenz-Landmine:** SMPL/SMPL-X-Modelldateien sind **research-only** (kommerziell =
kostenpflichtige Meshcapade-Lizenz). Kontaminiert fast alle 3D-Extraktoren. Ausweg für
kommerziell: nur Skelett-Keypoints/BVH intern, keine SMPL-Assets ausliefern; oder SAM 3D
Body + MHR (Apache-2.0). **Für privat kein Problem.**

**Qualitätsgrenzen:** Hände/Finger extrahieren aus Ganzkörper-Framing NICHT zuverlässig
(prozedurale Handanimation einplanen). Fuß-Sliding garantiert ohne Kontakt+IK-Pass.
60-fps-Quellen helfen deutlich.

---

## 5. Kamera-Kontrolle & Konsistenz

**Kamera (3 Stufen):**
1. **3D-Lift (Default):** Skelett aus virtueller Kamera neu rendern → deterministisch, volle Kontrolle.
2. **Generativ:** Wan2.2-Fun-Camera-Control (Presets), Civitai-Kamera-LoRAs (Orbit/Dolly).
3. **Re-Shoot:** ReCamMaster (MIT) auf fertige Clips; Uni3C (Apache-2.0) für 3D-konsistenten Hintergrund.

**Hintergrund-Problem:** 3D-Lift re-projiziert nur das Skelett, nicht die Szene. Lösungen:
Uni3C (Punktwolke), per-Winkel-Startbilder via Qwen-Image-Edit-2511 Multiple-Angles-LoRA
(96 Posen), Matrix-3D (MIT) für persistente Sets.

**Charakter-Konsistenz (nach Verlässlichkeit):**
1. Charakter-LoRA (musubi-tuner) + Multi-View-Sheet via Qwen-Image-Edit 2511 → mit **VACE** (nicht Animate).
2. Zero-shot: Phantom-Wan (Ganzkörper-Ref) oder Stand-In+VACE (Gesicht).
- Seeds konservieren KEINE Identität über Clips hinweg.

---

## 6. Choreographer

**Datenmodell:** Interne Repräsentation = SMPL-Pose-Parameter @ fixe FPS; pro Move-Asset:
Rotationen + Root-Trajektorie + Fußkontakte + Beat-Grid. Konverter an den Rändern (BVH/FBX-Export).

**v1 (deterministisch, kein ML):** Beat-getaggte Timeline, Moves an Downbeats segmentieren
(Beat This!), trim/loop/mirror/reverse/time-warp, Crossfade-Blend mit Foot-Lock, Root-Pfad-Editing,
3D-Vorschau (three.js/aitviewer). Alles Quaternion-Mathematik (fairmotion, BSD) — kann nicht halluzinieren.

**v1.5 (ML-Assists):** Übergangs-Glättung (CondMDI), "Remix" (SinMDM, MIT), Text-Edit (MotionReFit),
Style-Knopf (SMooDi), neue Moves aus Text (Kimodo / Tencent HY-Motion).

**Referenz:** **ComfyUI-Magos-Nodes** (GPL, Mai 2026) hat ~70 % des Choreographers bereits
gebaut (Timeline, Dope-Sheet, Graph-Editor, Orbit-Kamera, animierte Kamera-Keyframes) —
zur Konzept-Validierung nachbauen, nicht kopieren.

**Keine fertige Open-Source-"Choreographie-Timeline" existiert** — muss aus Bausteinen assembliert werden.

---

## 7. Post-Processing

| Tool | Lizenz | Rolle |
|---|---|---|
| **SeedVR2** (3B/7B) | Apache-2.0 | Upscale 480/720p → 1080p, temporal konsistent. **batch_size 4n+1, ≥5!** |
| **FlashVSR v1.1** | Apache-2.0 | Fast-Draft-Upscale (RTX-Support noch wackelig) |
| **Practical-RIFE** v4.25 | MIT | Frame-Interpolation 16→24/30/60 fps |
| **Real-ESRGAN** | BSD-3 | Instant-Draft-Upscale (flickert auf Diffusion-Footage) |

**Reihenfolge: erst Upscale, dann Interpolieren** (Diffusion-SR skaliert mit Framezahl).
VFI pro Shot, nie über Schnitte. Wan-2.2-720p gibt nativ 24 fps → VFI nur bei 16-fps-Quellen.
**Blocker (non-commercial):** GIMM-VFI, Upscale-A-Video, ComfyUI-Rife-Tensorrt-Node.

---

## 8. Audio-Pipeline

| Tool | Lizenz | Rolle |
|---|---|---|
| **Beat This!** (ohne --dbn) | MIT | Beat/Downbeat-Grid für Timeline-Snapping |
| **Kim Mel-Band RoFormer** | MIT | Musik/Vocal-Separation (nur noch optional, s. Audio-Entscheidung) |
| **Signalsmith Stretch** | MIT | Time-Stretch bei BPM-Anpassung (nur Feinabstimmung) |
| **FFmpeg** (BtbN lgpl-shared) | LGPL 2.1+ | Export-Muxing (MP4/AAC + WebM/Opus) |

**Sync-Kern:** Beat This! analysiert Quellvideo (Extraktion) UND eigenes Audiofile (Ziel-Raster).
Timeline snappt Moves auf neues Beat-Grid, Time-Warp auf Ziel-BPM (~±15 % natürliche Grenze).
madmom-Modelle NIE ausliefern (CC BY-NC-SA). libfdk_aac nie bündeln.

---

## 9. Architektur

**Engine:** ComfyUI als gemanagter Subprozess hinter eigener FastAPI-Orchestrierung
(Muster: Krita AI Diffusion, SwarmUI — hält eigenen Code GPL-frei). Eigener Job-Queue (SQLite),
Pipeline-DAG, WebSocket-Progress.

**Frontend:** Tauri v2 (oder Electron) + React + three.js-Viewport. Gradio nur für Prototypen.

**Installer:** ComfyUI-Desktop-Strategie kopieren (uv + gelockte Requirements + prebuilt
GPU-Wheels) — löst das #1-Windows-Problem (CUDA/pip-Breakage).

**Modell-Manager:** InvokeAI-Vorbild (Apache-2.0, Code wiederverwendbar) — Record/Install/
DownloadQueue-Services, ein geteiltes Modell-Verzeichnis (StabilityMatrix-Symlink-Muster).

**Draft-vs-Final:** Draft = 480p, GGUF Q4/Q5, 4-Step-Lightning-LoRA. Final = fp8/Q8, volle
Steps + Upscale + Interpolation. Artefakte content-hash-gecacht (Posen/SMPL/Control-Videos).

**VRAM-Stufen:** 12 GB = 480–576p GGUF Q4; 16 GB = fp8 + Block-Swap; 24–32 GB = 720p + LoRA-Training.

---

## 10. Rechtslage (DE/EU) — für spätere Kommerzialisierung

- **yt-dlp:** OLG Hamburg 11/2024 wertet YouTubes Rolling Cipher als TPM → nicht bündeln,
  Datei-Upload primär, Link-Ingest als Nutzer-Opt-in.
- **EU AI Act Art. 50** (ab **2.8.2026**): KI-Video maschinenlesbar markieren (C2PA + Watermark);
  Open-Source NICHT ausgenommen. c2pa-rs (MIT) im Export.
- **Musik:** Eigenes Audio (Entscheidung) → GEMA-Problem entfällt.
- **Choreographie:** §2(1) Nr. 3 UrhG (ohne Fixierungserfordernis) — Signature-Routinen können
  geschützt sein; generische Moves niedrig-riskant.
- **Persönlichkeitsrechte:** Risiko liegt beim Nutzer-Startbild (echtes Gesicht = Deepfake,
  Einwilligung nötig). Kennzeichnung ≠ Einwilligung. Consent-Hinweis beim Referenzbild-Upload.
- **NSFW:** NudeNet/CLIP-Filter auf Referenzbild + Output-Frames (harm reduction, umgehbar).

---

## Marktlücke & Positionierung

**Benchmark:** Kling 3.0 MC (Qualität), Viggle (Massenmarkt). Gegen die gewinnt man Mitte 2026
keinen reinen Qualitätsvergleich.

**Niemand bietet:** vollständig lokal/privat, Flatrate statt Credits, **editierbare
Motion-Timeline zwischen Extraktion und Generierung** (Choreographer-Alleinstellung),
Seed-Reproduzierbarkeit, Charakter-Template-Mixing. Kein kommerzielles Produkt hat den
Pipeline-Break Video → editierbares Skelett → Re-Render.

---

## Nächster Schritt: Prototyp-Spike

Riskanteste ungetestete Annahme validieren: GVHMR-Motion aus Tanzvideo extrahieren → Skelett
aus **anderem** Kamerawinkel rendern → durch Wan2.2-Animate mit Startbild. Läuft diese Kette
überzeugend, steht das ganze Konzept.
