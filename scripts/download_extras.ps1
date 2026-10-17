# Download the two licence-gated helper models from the StableProjectorz mirrors.
# facebook/dinov3-vitl16-pretrain-lvd1689m is manual-gated and briaai/RMBG-2.0 is
# auto-gated, so an anonymous HF token cannot fetch them. These zips are the exact
# same weights, re-hosted by TRELLIS.2-stableprojectorz.
param(
    [string]$Destination = "$PSScriptRoot\..\models",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$models = @(
    @{ Name = "dinov3";    Url = "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/dinov3.zip" },
    @{ Name = "RMBG-2.0";  Url = "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/RMBG-2.0.zip" }
)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

foreach ($model in $models) {
    $target = Join-Path $Destination $model.Name
    $marker = Join-Path $target "model.safetensors"
    if ((Test-Path $marker) -and -not $Force) {
        Write-Host "[extras] $($model.Name) already present, skipping"
        continue
    }
    $zip = Join-Path $Destination "$($model.Name).zip"
    Write-Host "[extras] downloading $($model.Name) -> $zip"
    curl.exe -L --fail --retry 6 --retry-delay 5 --retry-all-errors -C - -o $zip $model.Url
    if ($LASTEXITCODE -ne 0) { throw "download failed for $($model.Name)" }
    Write-Host "[extras] extracting $($model.Name)"
    Expand-Archive -LiteralPath $zip -DestinationPath $Destination -Force
    Remove-Item -LiteralPath $zip -Force
    if (-not (Test-Path $marker)) {
        throw "$($model.Name): expected $marker after extraction"
    }
    Write-Host "[extras] $($model.Name) ready"
}
