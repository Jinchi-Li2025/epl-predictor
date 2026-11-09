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
$BackupRoot = Join-Path $ProjectRoot "backups\next_steps_v3_$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

$ExactFiles = @(
    "app.py",
    "src\assets\build_player_registry.py",
    "src\features\build_squad_features.py",
    "src\features\build_squad_training_dataset.py",
    "src\evaluation\archive_squad_snapshot.py",
    "src\evaluation\squad_feature_challenger.py",
    "src\evaluation\llm_squad_advisor.py",
    "src\context\fetch_lineups_football_data.py",
    "src\context\apply_lineup_overrides.py",
    "src\pipeline\system_health_check.py",
    "src\pipeline\run_daily_update.py"
)

foreach ($RelativePath in $ExactFiles) {
    $Existing = Join-Path $ProjectRoot $RelativePath
    if (Test-Path $Existing) {
        $BackupFile = Join-Path $BackupRoot $RelativePath
        New-Item -ItemType Directory -Force -Path (Split-Path $BackupFile) | Out-Null
        Copy-Item $Existing $BackupFile -Force
    }
}

$TargetPages = Join-Path $ProjectRoot "pages"
$BackupPages = Join-Path $BackupRoot "pages"
New-Item -ItemType Directory -Force -Path $TargetPages | Out-Null
New-Item -ItemType Directory -Force -Path $BackupPages | Out-Null

foreach ($Pattern in @("1_*.py", "2_*.py", "3_*.py")) {
    Get-ChildItem -Path $TargetPages -Filter $Pattern -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $BackupPages $_.Name) -Force }
}

foreach ($Pattern in @("1_*.py", "2_*.py")) {
    Get-ChildItem -Path $TargetPages -Filter $Pattern -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
}

foreach ($Directory in @("src\assets", "src\features", "src\evaluation", "src\context", "src\pipeline", ".streamlit", "data\manual")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $Directory) | Out-Null
}

foreach ($RelativePath in @(
    "app.py",
    "src\assets\build_player_registry.py",
    "src\features\build_squad_features.py",
    "src\features\build_squad_training_dataset.py",
    "src\evaluation\archive_squad_snapshot.py",
    "src\evaluation\squad_feature_challenger.py",
    "src\evaluation\llm_squad_advisor.py",
    "src\context\fetch_lineups_football_data.py",
    "src\context\apply_lineup_overrides.py",
    "src\pipeline\system_health_check.py",
    "src\pipeline\run_daily_update.py",
    ".streamlit\config.toml"
)) {
    Copy-Item (Join-Path $SourceRoot $RelativePath) (Join-Path $ProjectRoot $RelativePath) -Force
}

$ManualLineupFile = Join-Path $ProjectRoot "data\manual\lineup_overrides.csv"
if (-not (Test-Path $ManualLineupFile)) {
    Copy-Item (Join-Path $SourceRoot "templates\lineup_overrides_template.csv") $ManualLineupFile -Force
}

$ModelCenterPage = Get-ChildItem -Path (Join-Path $SourceRoot "pages") -Filter "1_*.py" -File | Select-Object -First 1
$PlayerPage = Get-ChildItem -Path (Join-Path $SourceRoot "pages") -Filter "3_*.py" -File | Select-Object -First 1

if (-not $ModelCenterPage -or -not $PlayerPage) {
    throw "The patch pages are incomplete. Extract the ZIP again and retry."
}

Copy-Item $ModelCenterPage.FullName (Join-Path $TargetPages $ModelCenterPage.Name) -Force
Copy-Item $PlayerPage.FullName (Join-Path $TargetPages $PlayerPage.Name) -Force
$ModelCenterTarget = Join-Path "pages" $ModelCenterPage.Name
$PlayerPageTarget = Join-Path "pages" $PlayerPage.Name

Push-Location $ProjectRoot
try {
    python -m py_compile app.py
    python -m py_compile $ModelCenterTarget
    python -m py_compile $PlayerPageTarget
    python -m py_compile "src\assets\build_player_registry.py"
    python -m py_compile "src\features\build_squad_features.py"
    python -m py_compile "src\features\build_squad_training_dataset.py"
    python -m py_compile "src\evaluation\archive_squad_snapshot.py"
    python -m py_compile "src\evaluation\squad_feature_challenger.py"
    python -m py_compile "src\evaluation\llm_squad_advisor.py"
    python -m py_compile "src\context\fetch_lineups_football_data.py"
    python -m py_compile "src\context\apply_lineup_overrides.py"
    python -m py_compile "src\pipeline\system_health_check.py"
    python -m py_compile "src\pipeline\run_daily_update.py"
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "EPL Predictor v3 upgrade installed." -ForegroundColor Green
Write-Host "Backup: $BackupRoot"
Write-Host "Next commands:"
Write-Host "python src\assets\build_player_registry.py"
Write-Host "python src\features\build_squad_features.py"
Write-Host "python src\evaluation\archive_squad_snapshot.py"
Write-Host "python src\features\build_squad_training_dataset.py"
Write-Host "python src\evaluation\squad_feature_challenger.py"
Write-Host "python src\evaluation\llm_squad_advisor.py"
Write-Host "python src\pipeline\system_health_check.py"
Write-Host "streamlit cache clear"
Write-Host "streamlit run app.py"
