# Runs the Vite dev server on the fixed port (5173) that the backend's
# CORS config expects (see src/api/main.py FRONTEND_ORIGINS default).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location "$root\src\frontend"
try {
    npm run dev
}
finally {
    Pop-Location
}
