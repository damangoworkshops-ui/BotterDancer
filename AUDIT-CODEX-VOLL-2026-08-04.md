# Codex-Voll-Audit (Masterprompt) + Fix-Runde — 2026-08-04

Externes read-only-Audit des gesamten Projekts via `CODEX-AUDIT-PROMPT.md`
(4 HOCH, 7 MITTEL). Gegenverifikation: 11 parallele Fable-Prüfagenten, je einer
pro Finding, mit Live-System-Zugriff (read-only). **Ergebnis: 11/11 CONFIRMED** —
mehrere Audit-Fixvorschläge waren allerdings überdimensioniert oder fehlerhaft;
umgesetzt wurden die verifizierten Minimal-Fixes. Alle Fixes getestet
(13 Testfälle, s.u.).

## Findings → Umsetzung

| # | Finding | Fix (umgesetzt) |
|---|---|---|
| F1 HOCH | Tools steuerten die Desktop-App auf 8188 statt der Projekt-Instanz (live belegt); start_comfy verweigerte Start; `--check` prüfte den Server nie | **Neu `tools/comfy_target.py`**: Fingerprint (v0.22 + kein `--base-directory` in argv) + Auto-Discovery über 8188/8190. `submit_workflow` submittet nur noch an bestätigte Projekt-Instanz (expliziter Fremd-`--server` = harter Fehler), `--check` meldet Server-Status ehrlich. Dashboard: `/free`, `/interrupt`, History, Checks nur gegen Fingerprint-URL (15s-Cache); ohne Projekt-Instanz wird die Desktop-App explizit NICHT angesprochen. `start_comfy.ps1`: identitätsbewusster Guard, weicht bei Fremdbelegung auf 8190 aus, Log bleibt immer `comfyui_utf8.log` (repariert zugleich das tqdm-Progress-Parsing) |
| F2 HOCH | Kein Artefaktvertrag: 150/81-Überschuss nur Warnung; SavePoseKps meldet Dateien nie an die History → „ERFOLG (keine Dateien)" auch bei fehlendem/verkürztem Output | Überschuss ist jetzt FEHLER (`--force` = bewusster Override). Neue Postcondition: nach Erfolg wird die SavePoseKps-JSON selbst gesucht (Prefix + mtime ≥ Submit), Frame-Einträge gegen `frame_load_cap` geprüft (Toleranz cap−1), sonst Exit 1 |
| F3 HOCH | `SETUP DONE` bewies nichts: Patch-Warnung deterministisch tot (eigene Ausgabedatei erzeugt Dirty-Status), CPU-Torch reichte | Patch-Check via `git apply --reverse --check`; CUDA + sm_120 als asserts in der Verifikation; torch==2.11.0/torchvision==0.26.0 gepinnt. Bewusst OHNE `--no-deps` (bräche frische venv — Audit-Vorschlag war hier fehlerhaft; Masterprompt präzisiert) |
| F4 HOCH | Pose-Output wurde vor dem Rendern gelöscht → Fehlschlag mittendrin vernichtete gute Sätze, stale Sidecars | Beide Tools (`filter_pose_v2`, `reproject_camera`): Lockfile (O_EXCL) → Temp-Geschwisterdir → PNG-Zähl-Validierung → alter Ordner fällt erst jetzt (rmtree-Guard bleibt) → `os.rename` + Sidecars via `os.replace` |
| F5 MITTEL | Bis ~89 % synthetische Frames → trotzdem DONE/Exit 0 | `--max-synth-share` (Default 0,3): Gate VOR jedem Touch am alten Output, Abbruch Exit 2 |
| F6 MITTEL | Voll geclippte Kamera-Pose → schwarze Frames + DONE (kostet einen Wan-Render) | Sichtbarkeits-Gate: >20 % Frames mit <4 sichtbaren Joints → Exit 4 vor jedem Write; `frames_low_visibility` im Meta. Hinweis: mit gesunden Daten unerreichbar (Auto-Framing + 1m-Distanz-Floor) — Defense-in-Depth für kaputte Trajektorien, Logik standalone verifiziert |
| F7 MITTEL | cam120/180-draft aktiv trotz Draft-nur-0–90°-Regel | Alle 10 Presets tragen `angle`/`profile`; `load_preset` sperrt draft ≥120° hart (`--force` = Warnung) |
| F8 MITTEL | Dashboard-Baseline 10 s/it (falsch übertragen) → False-Positive-Alarme mit LLM-Schuldzuweisung | 15,7 s/it (gemessener Wert), Kommentar korrigiert. Bekannte Restlücke: Ein-Wert-Baseline alarmiert im Draft-Betrieb nie — workloadabhängige Baselines wären der Vollausbau |
| F9 MITTEL | RIFE: 32-fps-Input → doppelt lange Zeitlupe mit ERFOLG | Preflight leitet Erwartung aus dem Graph ab (`input_fps × multiplier == frame_rate`, ffprobe), Mismatch = FEHLER; `force_rate≠0` macht die Kette fps-invariant → Check entfällt korrekt |
| F10 MITTEL | `--force`-Rotation konnte Bundle halb zerstören; DACL-Fehler nur Warnung | Komplett-Backup nach `.bak` (atomar pro Datei) vor jedem Überschreiben; DACL fatal und VOR dem Key-Write |
| F11 MITTEL | Datei-Marker übersprang neue pytorch3d-Imports; Regex-/Compile-Lücken ohne Fehler-Exit | Marker-Skip entfernt (Idempotenz strukturell: gepatchte Imports sind eingerückt, `^`-Anker matcht nie erneut); Residualscan mit Exit 1 |

## Tests (alle grün, 04.08.)

Syntax: py_compile ×7, PSParser ×2. Funktional: Fingerprint-Discovery findet 8190;
Fremd-Submit geblockt; draft-120-Sperre (Exit 2) + cam60-draft erlaubt;
Surplus-150-PNG-Verzeichnis = FEHLER; RIFE 32-fps-Input = FEHLER, 16-fps = OK;
Cert-No-Op mit DACL Exit 0; Patcher 0 Residual Exit 0; start_comfy erkennt
Desktop als fremd + Projekt-Instanz als laufend; filter_pose transaktional
(frisch + Swap, Lock/Tmp aufgeräumt, Zähler identisch zur Historie 1+79);
reproject Normal-Lauf 80 PNGs + Sidecars; F6-Arithmetik beidseitig.

## Meta-Erkenntnisse

- Das externe Audit war faktisch exzellent (11/11 bestätigt, z.T. untertrieben),
  aber 3 von 11 Fix-Vorschlägen wären falsch/überdimensioniert umgesetzt worden
  (naives `--no-deps`, Marker-pro-Block, Versions-Verzeichnis+Lock fürs Cert-Tool).
  **Findings extern einkaufen, Fixes selbst verifizieren.**
- Offen aus dem Audit (bewusst): workloadabhängige Dashboard-Baselines;
  voller generischer Artefaktvertrag (ffprobe-Postcondition für alle
  Video-Outputs); die dokumentierten 51 Tests von 19.07. liegen nicht im
  Projekt (waren Session-Scratchpad) — Testdateien künftig einchecken.
- Empfehlung aus dem Audit, weiterhin offen: git-Repo fürs Projekt (Audit
  konnte Herkunft paralleler Änderungen nicht feststellen).
