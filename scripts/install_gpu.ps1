<#
.SYNOPSIS
    Install/configure CUDA-enabled PyTorch for the Nexa AI venv.

.DESCRIPTION
    Idempotent. Verifies the venv at C:\Nexa\.venv, detects whether the
    installed torch is a CUDA build, and (if not) installs torch 2.7.1
    with CUDA 12.8 wheels. The cu128 build is the earliest that ships
    sm_120 (Blackwell consumer) binaries required for RTX 50-series
    GPUs. Optionally installs bitsandbytes when NEXA_LOAD_IN_4BIT=true
    is set in .env, and nvidia-cublas-cu12 so the STT path's
    cublas64_12.dll lookup succeeds.

    Re-running this script is safe; it only reinstalls when needed.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_gpu.ps1
#>

[CmdletBinding()]
param(
    [string] $VenvPython = "C:\Nexa\.venv\Scripts\python.exe",
    [string] $TorchVersion = "2.7.1",
    [string] $CudaTag = "cu128"
)

$ErrorActionPreference = "Stop"

function Write-Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Test-CublasDll {
    param([string] $PythonExe)
    & $PythonExe -c @"
import ctypes, sys
try:
    ctypes.WinDLL('cublas64_12.dll')
    print('cublas64_12.dll: FOUND')
except OSError as exc:
    print(f'cublas64_12.dll: MISSING ({exc})')
    sys.exit(1)
"@
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Python venv not found at $VenvPython. Create it first: py -3.11 -m venv C:\Nexa\.venv"
}

Write-Section "Inspecting current install"
$probe = & $VenvPython -c @"
import sys
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_count={torch.cuda.device_count()}")
        print(f"device_name={torch.cuda.get_device_name(0)}")
        print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
except ImportError:
    print("torch: NOT INSTALLED")
"@

$torchInstalled = $probe -match '^torch=(\d+\.\d+\.\d+)'
$torchIsCuda = $probe -match '\+cu\d+'
$cudaAvailableNow = $probe -match 'cuda_available=True'

Write-Host ""
Write-Host "Probe output:"
Write-Host $probe

$needsTorchInstall = -not $torchInstalled -or -not $torchIsCuda -or -not $cudaAvailableNow

if ($needsTorchInstall) {
    Write-Section "Installing torch==$TorchVersion ($CudaTag) wheels"
    & $VenvPython -m pip install --upgrade pip | Out-Host
    & $VenvPython -m pip install "torch==$TorchVersion" --index-url "https://download.pytorch.org/whl/$CudaTag" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "torch install failed." }
} else {
    Write-Section "torch already looks good - skipping reinstall"
}

Write-Section "Installing nvidia-cublas-cu12 (for STT cublas64_12.dll)"
& $VenvPython -m pip install --upgrade "nvidia-cublas-cu12" | Out-Host

# Decide on bitsandbytes: only when the user has opted into 4-bit in .env
$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"
$wants4Bit = $false
if (Test-Path -LiteralPath $envFile) {
    $envLines = Get-Content -LiteralPath $envFile
    foreach ($line in $envLines) {
        if ($line -match '^\s*NEXA_LOAD_IN_4BIT\s*=\s*(true|1|yes|on)\s*$') {
            $wants4Bit = $true
            break
        }
    }
}

if ($wants4Bit) {
    Write-Section "Installing bitsandbytes (4-bit requested)"
    & $VenvPython -m pip install --upgrade "bitsandbytes" | Out-Host
} else {
    Write-Section "Skipping bitsandbytes (NEXA_LOAD_IN_4BIT not enabled)"
}

Write-Section "Final verification"
$final = & $VenvPython -c @"
import sys
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_name={torch.cuda.get_device_name(0)}")
        print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
except ImportError:
    print("torch: STILL NOT INSTALLED")
    sys.exit(1)
try:
    import bitsandbytes
    print(f"bitsandbytes={bitsandbytes.__version__}")
except ImportError:
    print("bitsandbytes: not installed")
"@

Write-Host $final

Write-Section "Checking cublas64_12.dll"
$hasCublas = Test-CublasDll -PythonExe $VenvPython
if (-not $hasCublas) {
    Write-Warning "cublas64_12.dll still not loadable. STT will fall back to CPU."
    Write-Warning "If you need GPU STT, verify CUDA 12 runtime is on PATH or set NEXA_STT_DEVICE=auto."
}

Write-Section "Done"
if ($cudaAvailableNow -or $final -match 'cuda_available=True') {
    Write-Host "GPU path ready. Start the API with:" -ForegroundColor Green
    Write-Host "  $VenvPython -m uvicorn api:app --host 127.0.0.1 --port 8000" -ForegroundColor Green
    Write-Host "Then verify with:" -ForegroundColor Green
    Write-Host "  Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5" -ForegroundColor Green
} else {
    Write-Warning "CUDA is still not available. Check your NVIDIA driver and run nvidia-smi."
}
