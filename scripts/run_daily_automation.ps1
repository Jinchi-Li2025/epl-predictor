param([string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent))

$ErrorActionPreference = "Stop"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python was not found: $Python"
}
Push-Location $ProjectRoot
try {
    & $Python "src\pipeline\run_daily_update.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python "src\pipeline\run_operational_checks.py"
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
