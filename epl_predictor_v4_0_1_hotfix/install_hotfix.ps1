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
$BackupRoot = Join-Path $ProjectRoot "backups\evidence_v4_0_1_$Timestamp"
$TargetScript = Join-Path $ProjectRoot "src\context\validate_llm_evidence.py"
$TargetOverride = Join-Path $ProjectRoot "data\manual\player_team_overrides.csv"

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $TargetScript) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $TargetOverride) | Out-Null

if (Test-Path $TargetScript) {
    Copy-Item $TargetScript (Join-Path $BackupRoot "validate_llm_evidence.py") -Force
}

Copy-Item (Join-Path $SourceRoot "src\context\validate_llm_evidence.py") $TargetScript -Force

if (-not (Test-Path $TargetOverride)) {
    Copy-Item (Join-Path $SourceRoot "templates\player_team_overrides_template.csv") $TargetOverride -Force
}

Push-Location $ProjectRoot
try {
    & python -m py_compile "src\context\validate_llm_evidence.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Python compilation failed."
    }

    foreach ($ScriptPath in @(
        "src\context\validate_llm_evidence.py",
        "src\reporting\build_match_briefs.py",
        "src\pipeline\system_health_check.py"
    )) {
        & python $ScriptPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python step failed: $ScriptPath"
        }
    }

    & python -m streamlit cache clear
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit cache clear failed."
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "EPL Predictor v4.0.1 evidence hotfix completed." -ForegroundColor Green
Write-Host "Backup: $BackupRoot"
Write-Host "Start the app with: python -m streamlit run app.py"
