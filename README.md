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
| `tools/` | Pipeline-Werkzeuge: `submit_workflow.py` (Preflight/Submit/Poll mit Artefaktvertrag), `reproject_camera.py` (3D→2D-Kamera-Reprojektion), `filter_pose_v2.py` (Pose-Isolation, transaktional), `export_clip.py` (Watermark + C2PA-Export), `dashboard.py` (System-HUD), `comfy_target.py` (Instanz-Fingerprint), Setup-Skripte |
| `workflows/` | ComfyUI-API-Workflows + Presets (ein Graph, Presets patchen) |
| `tests/` | Testsuite (synthetische Fixtures, keine persönlichen Medien) |
| `*.md` | Research, Architektur-Reviews, Audits mit Fix-Historie |

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
