# GVHMR-Setup (Stage 2) -- separates venv, Blackwell-tauglicher Torch-Override
# Kein $ErrorActionPreference=Stop: uv/pip schreiben Fortschritt auf stderr (PS 5.1 wuerde sterben).
# Stattdessen explizite Exit-Code-Checks.
# Gehaertet 2026-07-19 (Audit P0-5 + Re-Attack): uv-Check, Pfad-Guards, Shim-Restore
# unconditional, Patch-Check, Verifikation mit echtem Exit-Code.
# NUR ASCII in dieser Datei (PS 5.1 + BOM-loses UTF-8 = ParserError bei Em-Dash in Strings).
if (-not (Test-Path "C:\GVHMR\requirements.txt")) {
  Write-Output "FEHLER: C:\GVHMR nicht gefunden oder kein GVHMR-Checkout."
  exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  # Ohne diesen Check blieb $LASTEXITCODE vom letzten Fremdkommando stehen und
  # das Skript lief mit stillen Fehlschlaegen weiter.
  Write-Output "FEHLER: uv nicht im PATH (Installation: https://docs.astral.sh/uv/)."
  exit 1
}
Set-Location C:\GVHMR

# 1) Gefilterte Requirements (ohne torch-Pins, ohne Linux-pytorch3d-Wheel, numpy-Pin -> <2)
$req = Get-Content requirements.txt | Where-Object {
  $_ -notmatch '^(--extra-index-url|torch==|torchvision==|pytorch3d @)' -and
  $_ -notmatch '^numpy==' -and
  # cython_bbox braucht MSVC Build Tools (fehlen) und wird im GVHMR-Code nirgends importiert
  # (grep -r "cython_bbox" -> 0 Treffer ausser requirements.txt selbst).
  $_ -notmatch '^cython_bbox' -and
  # chumpy ist mit numpy<2 ohnehin unimportierbar (braucht numpy 1.23) und wird nur
  # fuer klassische SMPL-.pkl-Bodymodels gebraucht -- wir nutzen SMPLX-.npz.
  # Entfernt zugleich den --no-build-isolation-Sonderweg (chumpys defektes pyproject).
  $_ -notmatch '^chumpy'
}
$req += 'numpy<2'
$req | Set-Content requirements_win.txt -Encoding ascii

function Step($name, $script) {
  Write-Output ">>> $name"
  & $script 2>&1 | ForEach-Object { "$_" }
  if ($LASTEXITCODE -ne 0) { Write-Output "!!! FEHLGESCHLAGEN: $name (exit $LASTEXITCODE)"; exit 1 }
}

if (-not (Test-Path .\.venv\Scripts\python.exe)) { Step "venv" { uv venv --python 3.10 .venv } }
& .\.venv\Scripts\python.exe -c "import torch" 2>$null
$torchOk = ($LASTEXITCODE -eq 0)
if (-not $torchOk) {
  # Gepinnt auf den validierten Stand (04.08.2026, Audit F3): ungepinnt zog jede
  # Neuinstallation die jeweils neueste Version und konnte still vom getesteten
  # Setup abweichen. KEIN --no-deps hier: in einer frischen venv braucht torch
  # seine Import-Deps (filelock/sympy/...) -- die --no-deps-Regel gilt fuer
  # NACHinstallationen in die bestehende ComfyUI-venv, nicht fuer diesen Erstaufbau.
  Step "torch cu128" { uv pip install --python .\.venv\Scripts\python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 }
}
Step "pip fuer Builds" { uv pip install --python .\.venv\Scripts\python.exe pip setuptools wheel }
Step "requirements" { uv pip install --python .\.venv\Scripts\python.exe -r requirements_win.txt }
Step "gvhmr package" { uv pip install --python .\.venv\Scripts\python.exe -e . --no-deps }

# 2) pytorch3d-Shim UNCONDITIONAL wiederherstellen (Re-Attack-Befund: der alte
#    "nur wenn fehlt"-Guard liess einen veralteten Shim ohne ops\knn.py ewig liegen).
Write-Output ">>> pytorch3d-Shim kopieren (unconditional)"
$shimSrc = Join-Path $PSScriptRoot "pytorch3d_shim\pytorch3d"
Copy-Item -Recurse -Force $shimSrc ".\.venv\Lib\site-packages\"

# 3) Patch-Zustand KONKRET pruefen (Audit 04.08. F3): der alte Dirty-Status-Check
#    war deterministisch tot -- das Skript erzeugt mit requirements_win.txt in
#    jedem Lauf seinen eigenen Dirty-Status und unterdrueckte so die Warnung.
#    git apply --reverse --check: Exit 0 = Patch ist appliziert.
git -C C:\GVHMR apply --reverse --check "$PSScriptRoot\gvhmr_botterdancer.patch" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Output "!!! WARNUNG: BotterDancer-Patch ist NICHT appliziert (reverse-check schlug fehl)."
  Write-Output "    Anwenden mit:  git -C C:\GVHMR apply `"$PSScriptRoot\gvhmr_botterdancer.patch`""
  Write-Output "    Danach:        python `"$PSScriptRoot\patch_pytorch3d_optional.py`""
}

# 4) Verifikation MIT Exit-Code -- "SETUP DONE" nur, wenn sie wirklich besteht.
#    Audit 04.08. F3: CUDA + sm_120 sind jetzt ASSERTS, nicht nur Ausgabe --
#    ein CPU-Torch konnte vorher SETUP DONE melden und erst im GVHMR-Lauf sterben.
Write-Output ">>> Verifikation"
& .\.venv\Scripts\python.exe -c "import torch, pytorch3d.transforms, pytorch3d.ops.knn, hmr4d; assert torch.cuda.is_available(), 'CUDA nicht verfuegbar (CPU-Torch?)'; assert torch.cuda.get_device_capability(0) >= (12, 0), 'GPU ist kein sm_120 (Blackwell)'; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('cap', torch.cuda.get_device_capability(0))"
if ($LASTEXITCODE -ne 0) {
  Write-Output "!!! VERIFIKATION FEHLGESCHLAGEN (exit $LASTEXITCODE) -- Setup ist NICHT ok."
  exit 1
}
Write-Output "SETUP DONE"
