param(
    [switch]$PreflightOnly,
    [int]$Port = 8509
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = $env:OPENINTENT_PYTHON
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    $Fallback = Get-Command python -ErrorAction SilentlyContinue
    if ($Fallback) { $Python = $Fallback.Source }
}
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: no usable Python found." -ForegroundColor Red
    Write-Host "Do not reinstall PyTorch. From the original workspace use .venv, or set OPENINTENT_PYTHON to that python.exe."
    exit 1
}

$env:HF_HOME = Join-Path $Root "data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "=== OpenIntent classroom demo preflight ===" -ForegroundColor Cyan
& $Python (Join-Path $Root "preflight_demo.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "NOT READY: demo was not started. See preflight output above." -ForegroundColor Red
    exit $LASTEXITCODE
}
if ($PreflightOnly) {
    Write-Host "Preflight-only mode complete."
    exit 0
}

Write-Host "=== Starting Streamlit on http://127.0.0.1:$Port ===" -ForegroundColor Green
& $Python -m streamlit run (Join-Path $Root "app_demo.py") --server.address 127.0.0.1 --server.port $Port --server.headless true --server.fileWatcherType none
