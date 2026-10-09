param(
    [string]$FrontendPath = '',
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$backendPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $backendPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the backend .venv and install requirements.txt first.'
}
if (-not $FrontendPath) {
    $FrontendPath = & $pythonPath -c "import os, sys; from dotenv import dotenv_values; print(os.getenv('FRONTEND_PROJECT_DIR') or dotenv_values(sys.argv[1]).get('FRONTEND_PROJECT_DIR') or '')" (Join-Path $backendPath '.env')
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the local frontend configuration.' }
}
if (-not $FrontendPath) {
    throw 'Provide -FrontendPath or configure FRONTEND_PROJECT_DIR in local .env.'
}
$frontendDirectory = (Resolve-Path -LiteralPath $FrontendPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory 'package.json'))) {
    throw 'FrontendPath must point to the TutorFlow frontend repository.'
}

$variableNames = @('VITE_API_BASE_URL', 'PUBLIC_SHARE_MODE', 'FRONTEND_DIST_DIR')
$previousValues = @{}
foreach ($name in $variableNames) { $previousValues[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }

try {
    $env:VITE_API_BASE_URL = '/api'
    Push-Location -LiteralPath $frontendDirectory
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }

    $env:PUBLIC_SHARE_MODE = 'true'
    $env:FRONTEND_DIST_DIR = Join-Path $frontendDirectory 'dist'
    Write-Host "TutorFlow sharing mode: http://127.0.0.1:$Port"
    Write-Host 'Login credentials are SHARE_USERNAME and SHARE_PASSWORD in backend .env.'
    Push-Location -LiteralPath $backendPath
    try {
        & $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port
        if ($LASTEXITCODE -ne 0) { throw 'Backend stopped with an error. Check the credentials and whether the port is already in use.' }
    } finally { Pop-Location }
} finally {
    foreach ($name in $variableNames) { [Environment]::SetEnvironmentVariable($name, $previousValues[$name], 'Process') }
}
