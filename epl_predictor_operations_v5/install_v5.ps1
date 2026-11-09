param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$SourceRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

if (-not (Test-Path (Join-Path $ProjectRoot "app.py"))) {
    throw "The target is not the EPL Predictor project root: $ProjectRoot"
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $ProjectRoot "backups\operations_v5_$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

$Files = @(
    "src\evaluation\calibration_challenger.py",
    "src\evaluation\registry_quality_audit.py",
    "src\pipeline\pipeline_contract_audit.py",
    "src\pipeline\operational_readiness.py",
    "src\pipeline\run_matchday_refresh.py",
    "src\pipeline\run_operational_checks.py",
    "src\pipeline\run_daily_update.py",
    "src\pipeline\system_health_check.py",
    "scripts\run_daily_automation.ps1",
    "scripts\run_matchday_automation.ps1",
    "scripts\install_v5_windows_tasks.ps1",
    "scripts\uninstall_v5_windows_tasks.ps1"
)

foreach ($RelativePath in $Files) {
    $Target = Join-Path $ProjectRoot $RelativePath
    if (Test-Path $Target) {
        $Backup = Join-Path $BackupRoot $RelativePath
        New-Item -ItemType Directory -Force -Path (Split-Path $Backup) | Out-Null
        Copy-Item $Target $Backup -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
    Copy-Item (Join-Path $SourceRoot $RelativePath) $Target -Force
}

$OperationsPage = Get-ChildItem -Path (Join-Path $SourceRoot "pages") -Filter "4_*.py" -File | Select-Object -First 1
if (-not $OperationsPage) {
    throw "The operations page is missing. Extract the ZIP again."
}
$TargetPage = Join-Path (Join-Path $ProjectRoot "pages") $OperationsPage.Name
New-Item -ItemType Directory -Force -Path (Split-Path $TargetPage) | Out-Null
if (Test-Path $TargetPage) {
    Copy-Item $TargetPage (Join-Path $BackupRoot $OperationsPage.Name) -Force
}
Copy-Item $OperationsPage.FullName $TargetPage -Force

Push-Location $ProjectRoot
try {
    foreach ($RelativePath in $Files) {
        if ($RelativePath.EndsWith(".py")) {
            & python -m py_compile $RelativePath
            if ($LASTEXITCODE -ne 0) {
                throw "Python compilation failed: $RelativePath"
            }
        }
    }
    & python -m py_compile $TargetPage
    if ($LASTEXITCODE -ne 0) {
        throw "Python compilation failed: $($OperationsPage.Name)"
    }
    & python "src\pipeline\run_operational_checks.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Initial operational checks failed."
    }
    & python -m streamlit cache clear
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit cache clear failed."
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "EPL Predictor v5 operations and shadow validation installed." -ForegroundColor Green
Write-Host "Backup: $BackupRoot"
Write-Host "Start the app with: python -m streamlit run app.py"
Write-Host "Optional Windows tasks are NOT installed automatically."
Write-Host "To install them later: powershell -ExecutionPolicy Bypass -File .\scripts\install_v5_windows_tasks.ps1"
