# BotterDancer -- ComfyUI-Start mit den Perf-Fixes aus dem Deep-Review (2026-07-10)
# Aenderungen ggue. altem Start:
#   - KEIN --highvram   (darunter ist offload_device=cuda -> /free wird zum VRAM-No-Op)
#   - --reserve-vram 4  (Puffer fuer Fremdprozesse/WDDM)
#   - Log als UTF-8 via cmd-Redirect (PowerShell *> schreibt UTF-16LE -> grep-blind)
# Fixes 2026-07-19 (Audit A5 + Review + Re-Attack):
#   - Doppelstart-Guard; Vorgaenger-Log wird datumsgestempelt archiviert statt vernichtet.
#   - NUR ASCII in dieser Datei: PS 5.1 liest BOM-loses UTF-8 als ANSI; ein Em-Dash
#     wird dann zu einem Smart-Quote, das Strings SCHLIESST -> ParserError vor Zeile 1.
# Fix 2026-08-04 (Voll-Audit F1): Der alte Guard hielt JEDEN 8188-Listener fuer die
# Projekt-Instanz -- seit die ComfyUI-Desktop-App (v0.30, eigene Pfade) den Port
# belegt, verweigerte das Skript den Start komplett. Jetzt: Identitaetspruefung
# per /system_stats-Fingerprint; bei Fremdbelegung Ausweichen auf 8190.
# Log bleibt IMMER comfyui_utf8.log (submit_workflow/dashboard lesen daraus).

function Test-ProjectInstance($port) {
  # 1 = Projekt-Instanz, 0 = fremd, -1 = Port frei/nicht antwortend
  if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    return -1
  }
  $stats = $null
  try { $stats = Invoke-RestMethod "http://127.0.0.1:$port/system_stats" -TimeoutSec 4 } catch { return 0 }
  $ver = "$($stats.system.comfyui_version)"
  $argvJoined = ""
  if ($stats.system.argv) { $argvJoined = ($stats.system.argv -join " ") }
  if ($ver -like "0.22*" -and $argvJoined -notlike "*--base-directory*") { return 1 }
  return 0
}

if (-not (Test-Path "C:\ComfyUI\venv\Scripts\python.exe")) {
  Write-Output "FEHLER: C:\ComfyUI\venv\Scripts\python.exe nicht gefunden."
  exit 1
}

$Port = 0
foreach ($cand in @(8188, 8190)) {
  $state = Test-ProjectInstance $cand
  if ($state -eq 1) {
    Write-Output "Projekt-ComfyUI laeuft bereits auf Port $cand -- kein Doppelstart."
    exit 1
  }
  if ($state -eq 0) {
    Write-Output "Port $cand ist von einer FREMDEN Instanz belegt (vermutlich ComfyUI-Desktop) -- ueberspringe."
    continue
  }
  $Port = $cand
  break
}
if ($Port -eq 0) {
  Write-Output "FEHLER: Kein freier Kandidaten-Port (8188/8190) -- Fremdinstanzen beenden oder Port freigeben."
  exit 1
}

$log = "C:\ComfyUI\comfyui_utf8.log"
if (Test-Path $log) {
  $stamp = (Get-Item $log).LastWriteTime.ToString("yyyyMMdd_HHmmss")
  $dst = "C:\ComfyUI\comfyui_utf8_$stamp.log"
  $n = 2
  while (Test-Path $dst) { $dst = "C:\ComfyUI\comfyui_utf8_${stamp}_$n.log"; $n++ }
  Move-Item $log $dst
  if (Test-Path $log) {
    Write-Output "FEHLER: Log-Archivierung fehlgeschlagen ($log haengt fest?) -- Abbruch statt Truncate."
    exit 1
  }
}
Write-Output "Starte Projekt-ComfyUI auf Port $Port (Log: $log)"
Set-Location C:\ComfyUI
cmd /c "venv\Scripts\python.exe main.py --listen 127.0.0.1 --port $Port --reserve-vram 4 > `"$log`" 2>&1"
