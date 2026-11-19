param(
    [switch]$PreflightOnly,
    [int]$Port = 8506
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ApiFile = Join-Path $Root "configs\demo_api.env"
if (-not (Test-Path $ApiFile)) {
    Write-Host "ERROR: configs\demo_api.env is missing." -ForegroundColor Red
    Write-Host "Copy configs\demo_api.env.example, fill in the key locally, and run this script again."
    Write-Host "The key is intentionally not committed to Git."
    exit 1
}

# Load only the four DeepSeek settings. Never print the key.
$Loaded = 0
Get-Content -LiteralPath $ApiFile -Encoding utf8 | ForEach-Object {
    if ($_ -match '^\s*(DEEPSEEK_[A-Z_]+)\s*=\s*(.*?)\s*$') {
        $name = $matches[1]
        $value = $matches[2].Trim().Trim('"').Trim("'")
        if ($name -eq "DEEPSEEK_API_KEY" -and ($value -eq "" -or $value -like "sk-your-key-here")) {
            Write-Host "ERROR: DEEPSEEK_API_KEY is empty or still the example placeholder." -ForegroundColor Red
            exit 1
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
        $Loaded++
    }
}
if ($Loaded -lt 2) {
    Write-Host "ERROR: configs\demo_api.env did not contain the expected DEEPSEEK_* values." -ForegroundColor Red
    exit 1
}

$Python = $env:OPENINTENT_PYTHON
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) {
    $Fallback = Get-Command python -ErrorAction SilentlyContinue
    if ($Fallback) { $Python = $Fallback.Source }
}
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: no usable Python found. Set OPENINTENT_PYTHON to the original .venv python." -ForegroundColor Red
    exit 1
}

$env:HF_HOME = Join-Path $Root "data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OPENINTENT_DEFAULT_MODE = "DeepSeek-assisted"
$env:OPENINTENT_DEFAULT_ROUTER = "Qwen-Plain42-RejectOnly"

Write-Host "=== Full rich demo preflight ===" -ForegroundColor Cyan
& $Python (Join-Path $Root "preflight_demo.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "NOT READY: rich demo was not started." -ForegroundColor Red
    exit $LASTEXITCODE
}
if ($PreflightOnly) { Write-Host "Preflight-only complete."; exit 0 }

Write-Host "=== Full rich demo: http://127.0.0.1:$Port ===" -ForegroundColor Green
Write-Host "Model: Qwen-Plain42-RejectOnly | default mode: DeepSeek-assisted"
& $Python -m streamlit run (Join-Path $Root "app_gate8.py") --server.address 127.0.0.1 --server.port $Port --server.headless true --server.fileWatcherType none
