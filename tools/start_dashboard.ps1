# BotterDancer HUD starten — http://127.0.0.1:8189
# Nutzt das ComfyUI-venv-Python (stdlib-only, keine Zusatzpakete noetig).
$py = "C:\ComfyUI\venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  # Blindes "python" liefe auf Win 11 in den Microsoft-Store-Alias-Stub.
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if (-not $cmd -or $cmd.Source -like "*WindowsApps*") {
    Write-Output "FEHLER: ComfyUI-venv nicht gefunden ($py) und kein echtes python im PATH."
    exit 1
  }
  $py = $cmd.Source
}
& $py "$PSScriptRoot\dashboard.py"
