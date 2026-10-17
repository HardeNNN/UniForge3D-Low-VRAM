# Windows/CUDA installer for UniForge3D on consumer GPUs (8 GB tested on an RTX 3070 Ti).
#
# Differences from upstream setup.sh:
#   * No conda, no source builds. Every CUDA extension comes from a prebuilt
#     cp311 win_amd64 wheel (whl/).
#   * Python 3.11 is required to match those wheels.
#   * flash-attn comes from a prebuilt Windows wheel instead of a 30 minute
#     nvcc build.
param(
    [string]$VenvPath = "$PSScriptRoot\..\.venv",
    [switch]$WithoutFlashAttn,
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$whl = Join-Path $root "whl"
$python = Join-Path $VenvPath "Scripts\python.exe"

function Invoke-Pip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Write-Host "[install] pip $($Args -join ' ')"
    & $python -m pip install --no-input --retries 15 --timeout 60 @Args
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($Args -join ' ')" }
}

if (-not (Test-Path $python)) {
    $launcher = $null
    foreach ($candidate in @(
            "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
            "C:\Program Files\Python311\python.exe")) {
        if (Test-Path $candidate) { $launcher = $candidate; break }
    }
    if (-not $launcher) { $launcher = "py -3.11" }
    Write-Host "[install] creating virtual environment at $VenvPath"
    if ($launcher -eq "py -3.11") {
        py -3.11 -m venv $VenvPath
    } else {
        & $launcher -m venv $VenvPath
    }
    if (-not (Test-Path $python)) { throw "Python 3.11 not found; install it first (winget install Python.Python.3.11)" }
}

& $python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
Invoke-Pip --upgrade pip setuptools wheel

# 1. PyTorch 2.8.0 + CUDA 12.8: the ABI every prebuilt wheel in whl/ was built against.
# Local copies win: scripts/fetch.py downloads them resumably over many connections,
# while pip's own client stalled on this 3.4 GB file.
if (-not $SkipTorch) {
    $torchWheel = Get-ChildItem (Join-Path $whl "torch-*.whl") -ErrorAction SilentlyContinue | Select-Object -First 1
    $visionWheel = Get-ChildItem (Join-Path $whl "torchvision-*.whl") -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($torchWheel -and $visionWheel) {
        Invoke-Pip $torchWheel.FullName $visionWheel.FullName
    } else {
        Write-Host "[install] local torch wheels missing; fetching from the PyTorch index"
        Invoke-Pip torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    }
}

# 2. huggingface-hub is pinned before everything else: hub 1.x changed the client API
# that transformers 4.57 uses for local `from_pretrained`. The weight downloader
# (scripts/fetch.py) talks plain HTTP, so it does not need the newer client.
Invoke-Pip "huggingface-hub>=0.36,<1"

# 3. Pure-python dependencies (versions follow upstream setup.sh).
Invoke-Pip "setuptools==80.9.0" "wheel" "ninja" "packaging" `
    "transformers==4.57.3" "peft==0.17.1" `
    "timm>=1.0,<2" "kornia>=0.8,<0.9" "numpy>=2.1,<2.3" `
    "Pillow>=10.4,<12" "safetensors>=0.7,<1" "trimesh==4.6.3" `
    "opencv-python-headless>=4.10,<5" "easydict>=1.13,<2" `
    "tqdm>=4.66,<5" "scipy>=1.13,<2"

# 4. Prebuilt CUDA wheels (FlexGEMM needs Triton; the Windows port is a separate package).
Invoke-Pip "triton-windows==3.4.0.post21"

# 5. flash-attn from the prebuilt wheel.
if (-not $WithoutFlashAttn) {
    $flashAttn = Join-Path $whl "flash_attn-2.8.2+cu128torch2.8-cp311-cp311-win_amd64.whl"
    if (-not (Test-Path $flashAttn)) {
        New-Item -ItemType Directory -Force -Path $whl | Out-Null
        $url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2%2Bcu128torch2.8-cp311-cp311-win_amd64.whl"
        Write-Host "[install] downloading flash-attn wheel"
        curl.exe -L --fail --retry 6 --retry-delay 5 --retry-all-errors -C - -o $flashAttn $url
        if ($LASTEXITCODE -ne 0) { throw "flash-attn wheel download failed" }
    }
    Invoke-Pip $flashAttn
}

# 6. The rest of the local CUDA wheels (cumesh, flex_gemm, o_voxel, nvdiffrast, ...).
foreach ($wheel in Get-ChildItem (Join-Path $whl "*.whl") | Sort-Object Name) {
    # torch/torchvision are installed by step 1, flash-attn above.
    # Pillow-SIMD conflicts with the Pillow pin upstream declares and only speeds up
    # image ops, which are not the bottleneck here.
    if ($wheel.Name -like "torch*" -or $wheel.Name -like "flash_attn*" -or $wheel.Name -like "Pillow_SIMD*") { continue }
    Invoke-Pip $wheel.FullName
}

# 7. The web UI (webui.py). Pure python, no CUDA build involved.
Invoke-Pip "gradio>=5,<6"

Write-Host "[install] done. Verify with: $python scripts\check_env.py"
Write-Host "[install] then start the web UI with start_webui.bat (start_webui_lowram.bat for a machine with less than 64 GB of RAM)"
Write-Host "[install] optional: convert_weights_fp16.bat halves the fp32 checkpoints once and saves about 10 GB of disk"
