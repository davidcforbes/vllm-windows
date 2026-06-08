<#
.SYNOPSIS
    Serve BAAI/bge-m3 as an embedding endpoint on the native Windows vLLM (no Docker).

.DESCRIPTION
    Endpoint: http://localhost:<Port>/v1/embeddings   (default Port 8001)
    Model name (--served-model-name): bge-m3
    Native 1024-dim dense embeddings; max input 8192 tokens. Small (~2 GB VRAM),
    so it can share the GPU with other small workloads. Serves in the foreground;
    Ctrl-C to stop.

.NOTES
    Run examples:
      pwsh .\launch\serve_bge_m3.ps1
      pwsh .\launch\serve_bge_m3.ps1 -Port 8001 -GpuMemUtil 0.20
#>

[CmdletBinding()]
param(
    [string]$Model      = 'BAAI/bge-m3',
    [string]$ServedName = 'bge-m3',
    [int]$Port          = 8001,
    [string]$VllmExe    = 'C:\dev\vllm-windows\.venv\Scripts\vllm.exe',
    [string]$CudaBin    = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin',
    [double]$GpuMemUtil = 0.20,
    [int]$MaxModelLen   = 8192
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $VllmExe)) { Write-Error "vllm.exe not found at $VllmExe"; exit 2 }

$env:PATH = "$CudaBin;$env:PATH"
$env:PYTHONUTF8 = '1'
Set-Location $env:TEMP   # avoid source-shadowing the installed vllm._C

$vllmArgs = @(
    'serve', $Model,
    '--served-model-name', $ServedName,
    '--runner', 'pooling',
    '--convert', 'embed',
    '--dtype', 'auto',
    '--max-model-len', "$MaxModelLen",
    '--gpu-memory-utilization', "$GpuMemUtil",
    '--enforce-eager',
    '--host', '127.0.0.1',
    '--port', "$Port"
)

Write-Host "==> serving $Model as '$ServedName' on http://127.0.0.1:$Port/v1 (embeddings, native)" -ForegroundColor Cyan
& $VllmExe @vllmArgs
exit $LASTEXITCODE
