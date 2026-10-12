param([switch]$IfEnabled)

$ErrorActionPreference = 'Stop'
$backendDirectory = Split-Path -Parent $PSScriptRoot
$backendPython = Join-Path $backendDirectory '.venv\Scripts\python.exe'
Push-Location -LiteralPath $backendDirectory
try {
    $configuration = & $backendPython -c "import json; from app.config import ImageSettings; s=ImageSettings.from_environment(); print(json.dumps({'enabled':s.enabled,'url':s.comfy_url,'model':s.model}))"
    if ($LASTEXITCODE -ne 0) { throw 'Could not read local picture settings.' }
} finally { Pop-Location }
$configuration = $configuration | ConvertFrom-Json
if ($IfEnabled -and -not $configuration.enabled) { return }
$serviceUri = [Uri]$configuration.url

function Test-PictureService {
    try {
        $state = Invoke-RestMethod -Uri "$($configuration.url)/system_stats" -TimeoutSec 2
        return [bool]$state.system.comfyui_version
    } catch { return $false }
}

if (Test-PictureService) {
    Write-Host 'Local picture service is ready.'
    return
}
if (Get-NetTCPConnection -State Listen -LocalPort $serviceUri.Port -ErrorAction SilentlyContinue) {
    throw 'The configured picture-service port belongs to another service.'
}

$runtimeDirectory = Join-Path $backendDirectory 'data\image-runtime'
$comfyDirectory = Join-Path $runtimeDirectory 'ComfyUI'
$comfyPython = Join-Path $runtimeDirectory 'venv\Scripts\python.exe'
foreach ($requiredPath in @($comfyPython, (Join-Path $comfyDirectory 'main.py'))) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw 'Install ComfyUI in data/image-runtime with a separate venv first.' }
}
$diffusionModel = if ($configuration.model -eq 'flux2-klein-4b') { 'flux-2-klein-4b-fp8.safetensors' } else { 'z_image_turbo_bf16.safetensors' }
$vaeModel = if ($configuration.model -eq 'flux2-klein-4b') { 'flux2-vae.safetensors' } else { 'ae.safetensors' }
foreach ($modelPath in @("diffusion_models\$diffusionModel", 'text_encoders\qwen_3_4b.safetensors', "vae\$vaeModel")) {
    if (-not (Test-Path -LiteralPath (Join-Path $comfyDirectory "models\$modelPath"))) { throw 'A required ComfyUI model file is missing.' }
}

$listenAddress = if ($serviceUri.DnsSafeHost -eq '::1') { '::1' } else { '127.0.0.1' }
$service = Start-Process -FilePath $comfyPython -ArgumentList @(
    'main.py', '--listen', $listenAddress, '--port', $serviceUri.Port,
    '--disable-all-custom-nodes', '--offline', '--disable-auto-launch', '--disable-metadata', '--cache-none'
) -WorkingDirectory $comfyDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDirectory 'comfy.stdout.log') `
    -RedirectStandardError (Join-Path $runtimeDirectory 'comfy.stderr.log')
$deadline = [DateTime]::UtcNow.AddSeconds(45)
while ([DateTime]::UtcNow -lt $deadline) {
    if (Test-PictureService) {
        Write-Host 'Local picture service started on loopback.'
        return
    }
    Start-Sleep -Milliseconds 300
}
throw 'ComfyUI did not start. Check its local logs in data/image-runtime.'
