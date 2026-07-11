# One-time environment setup: Python venv + backend deps + frontend deps.
# Run once after cloning (or whenever requirements.txt / package.json change).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path "$root\.venv")) {
    py -3.12 -m venv "$root\.venv"
}

& "$root\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$root\.venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt"

Push-Location "$root\src\frontend"
npm install
Pop-Location

Write-Host "`nSetup complete. Start the app with:"
Write-Host "  .\scripts\run_backend.ps1   (in one terminal)"
Write-Host "  .\scripts\run_frontend.ps1  (in another terminal)"
