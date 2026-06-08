<#
.SYNOPSIS
    Serve a Gemma 12B (4-bit AWQ) chat/reasoning model on the native Windows vLLM (no Docker).

.DESCRIPTION
    Endpoint: http://localhost:<Port>/v1/chat/completions  (default Port 8002)
    Model name (--served-model-name): gemma
    Reserved for LLM inference/reasoning. A 12B model at 4-bit AWQ is ~6.5-7 GB of
    weights and fits the 16 GB GPU comfortably *on its own* (keep embeddings on
    CPU/ONNX and don't co-run Chandra). Serves in the foreground; Ctrl-C to stop.

.PARAMETER Model
    REQUIRED. HuggingFace id or local path of the 4-bit AWQ checkpoint. There is no
    verified official "Gemma 4 12B" AWQ, so point this at a real AWQ/GPTQ build,
    e.g. a community AWQ of Gemma-3-12B, or your own quantized export. Examples:
      -Model 'C:\models\gemma-12b-it-awq'
      -Model 'some-org/gemma-3-12b-it-AWQ'

.NOTES
    Run:
      pwsh .\launch\serve_gemma.ps1 -Model <awq-repo-or-path>
      pwsh .\launch\serve_gemma.ps1 -Model <...> -MaxModelLen 32768 -Cudagraphs
#>

[CmdletBinding()]
param(
    [string]$Model            = '',
    [string]$ServedName       = 'gemma',
    [int]$Port                = 8002,
    [string]$VllmExe          = 'C:\dev\vllm-windows\.venv\Scripts\vllm.exe',
    [string]$CudaBin          = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin',
    [string]$Quantization     = 'awq_marlin',
    [double]$GpuMemUtil       = 0.85,
    [int]$MaxModelLen         = 16384,
    [string]$KvCacheDtype     = 'fp8',
    [switch]$EnforceEager     = $false
)

$ErrorActionPreference = 'Stop'
if (-not $Model) {
    Write-Error ("Set -Model to your Gemma 12B AWQ checkpoint (HF id or local path). " +
        "No official 'Gemma 4 12B' AWQ was confirmed; use a real AWQ/GPTQ build, " +
        "e.g. -Model 'org/gemma-3-12b-it-AWQ' or a local C:\models\... path.")
    exit 2
}
if (-not (Test-Path $VllmExe)) { Write-Error "vllm.exe not found at $VllmExe"; exit 2 }

$env:PATH = "$CudaBin;$env:PATH"
$env:VLLM_USE_FLASHINFER_SAMPLER = '0'   # FlashInfer's sampler JIT fails on Windows
$env:PYTHONUTF8 = '1'
Set-Location $env:TEMP   # avoid source-shadowing the installed vllm._C

$vllmArgs = @(
    'serve', $Model,
    '--served-model-name', $ServedName,
    '--quantization', $Quantization,
    '--dtype', 'auto',
    '--max-model-len', "$MaxModelLen",
    '--kv-cache-dtype', $KvCacheDtype,
    '--gpu-memory-utilization', "$GpuMemUtil",
    # cudagraphs (default on here) speed up decode; Gemma alone has the VRAM.
    # Pass -EnforceEager to save ~1 GB / skip graph capture if you need headroom.
    $(if ($EnforceEager) { '--enforce-eager' } else { '--no-enforce-eager' }),
    '--host', '127.0.0.1',
    '--port', "$Port"
)

Write-Host "==> serving $Model as '$ServedName' on http://127.0.0.1:$Port/v1 (AWQ, native)" -ForegroundColor Cyan
& $VllmExe @vllmArgs
exit $LASTEXITCODE
