# BotterDancer

*(English: personal local-first dance motion-transfer pipeline for Windows —
tooling, workflows and audit docs. German project language below.)*

Lokale Windows-Pipeline für Dance-Motion-Transfer: Tanzvideo rein → 3D-Motion-
Extraktion (GVHMR) → KI-Charakter (Wan2.2-Animate via ComfyUI) tanzt die
Choreografie aus frei wählbaren Kamerawinkeln. Privates Einzelplatz-Projekt,
kein Produkt — dieses Repo enthält die **Werkzeuge, Workflows und die
Projekt-Dokumentation**, nicht die Medien-Assets.

## Was hier liegt

| Pfad | Inhalt |
|---|---|
| `tools/pipeline.py` | **Der Einstieg:** fährt aus einem Job-Rezept die ganze Kette — Segment → Beats → Pose → Render(s) → Compositing → Interpolation → signierter Export |
| `jobs/` | Job-Rezepte (JSON) als Vorlagen für die drei Betriebsarten |
| `tools/` | Werkzeugkasten: `crew_pose.py` (Multi-Person-Tracking), `crew_composite.py` (Freistellen ohne Greenscreen), `room_plate.py` (Raum ohne Personen), `motion_warp.py` (Choreo aufs Beat-Grid), `foot_anchor.py` (Foot-Skate), `reproject_camera.py` (virtuelle Kamerafahrten), `song_beats.py` (Beat This! + Segmentwahl), `export_clip.py` (Watermark + C2PA), `dashboard.py` (HUD), `seam_check.py`, `submit_workflow.py`, … |
| `workflows/` | ComfyUI-API-Workflows + Presets (ein Graph, Presets patchen) |
| `tests/` | 126 Tests, synthetische Fixtures, keine persönlichen Medien |
| `*.md` | Research, Architektur-Reviews, Audits mit Fix-Historie |

## Die drei Betriebsarten

Ein Job-Rezept wählt zwei Achsen — die Kopplung dazwischen ist echt, nicht bequem:

| Kamera | Hintergrund | Was geht | Was nicht |
|---|---|---|---|
| `static` | `studio` | Gruppen (Full-Crew, je Figur eigene Referenz), Original-Kameraführung | keine freie Kamera |
| `static` | `room` | dasselbe, aber im **echten Raum** des Quellclips | Plate braucht statische Kamera |
| `moving` | `studio` | frei geführte Kamerafahrt (Steadicam/Gimbal/Crane) + Beat-Warp + Foot-Anchoring | **nur eine Figur** (GVHMR trackt eine Person) |

```
C:\ComfyUI\venv\Scripts\python.exe tools\pipeline.py --job jobs\beispiel_static_crew.json
```

`--dry-run` zeigt den Plan, `--stop-after <schritt>` hält an einer Stufe an. Jeder
Lauf schreibt ein Job-Protokoll neben den Output.

**Draft-Falle:** `quality: "draft"` läuft mit cfg 1.0 — dort werden Negativ-Prompts
technisch nicht ausgewertet, das Modell erfindet dann teils zusätzliche Figuren.
Draft ist für Timing und Komposition; Figurenanzahl und Artefakte am Final beurteilen.

## Was hier bewusst NICHT liegt

- `assets/` — Treibervideos (zeigen echte Tänzer; Urheberrecht an Video und
  Choreografie), daraus abgeleitete Motion-Daten, Referenzbilder.
- `assets/keys/` — privates C2PA-Signaturmaterial. Wird lokal mit
  `tools/make_c2pa_cert.py` erzeugt.

## Kontext

Die Maschine dahinter: Windows 11, RTX PRO 6000 Blackwell (96 GB VRAM),
ComfyUI unter `C:\ComfyUI`, GVHMR unter `C:\GVHMR` — viele Pfade sind bewusst
hart kodiert, das ist ein Einzelplatz-Werkzeugkasten, keine portable Library.
Die Doku (insb. `GO-CHECKLIST.md`, `AUDIT-*.md`, `RESEARCH.md`) dokumentiert
Entscheidungen, Fehlschläge und Fixes chronologisch — sie ist als
Arbeitsjournal gedacht und vermutlich der interessanteste Teil des Repos.

Exporte tragen ein unsichtbares Watermark und ein C2PA-Manifest
(KI-Kennzeichnung, `trainedAlgorithmicMedia`) — siehe `tools/export_clip.py`.

## Tests

```
C:\ComfyUI\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ein Teil der Tests braucht die lokale Umgebung (ComfyUI-venv, ffmpeg);
rein logische Tests laufen mit jedem Python 3.10+.

## Lizenz

MIT (siehe `LICENSE`). `tools/pytorch3d_shim/` enthält unveränderte Teile von
[PyTorch3D](https://github.com/facebookresearch/pytorch3d) (BSD-3-Clause,
Copyright Meta Platforms — Lizenztext liegt im Ordner).
