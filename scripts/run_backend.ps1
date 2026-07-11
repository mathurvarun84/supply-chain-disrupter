# Runs the FastAPI backend on the fixed port (8173) that the frontend's
# Vite dev proxy expects (see src/frontend/vite.config.ts). Must run with
# cwd = project root since db_utils/etl_loader/yaml_utils use relative paths.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
try {
    & "$root\.venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8173 --reload
}
finally {
    Pop-Location
}
