param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

function Invoke-CheckedPython {
    param([string]$ScriptPath)
    & python $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed: $ScriptPath"
    }
}

& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "install_upgrade.ps1") -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Push-Location $ProjectRoot
try {
    Invoke-CheckedPython "src\assets\build_player_registry.py"
    Invoke-CheckedPython "src\features\build_squad_features.py"
    Invoke-CheckedPython "src\evaluation\archive_squad_snapshot.py"
    Invoke-CheckedPython "src\features\build_squad_training_dataset.py"
    Invoke-CheckedPython "src\evaluation\squad_feature_challenger.py"
    Invoke-CheckedPython "src\evaluation\llm_squad_advisor.py"
    Invoke-CheckedPython "src\context\validate_llm_evidence.py"
    Invoke-CheckedPython "src\reporting\build_match_briefs.py"
    Invoke-CheckedPython "src\evaluation\model_monitor.py"
    Invoke-CheckedPython "src\pipeline\system_health_check.py"
    & python -m streamlit cache clear
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit cache clear failed."
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Install and initial build completed." -ForegroundColor Green
Write-Host "Start the app with: python -m streamlit run app.py"
