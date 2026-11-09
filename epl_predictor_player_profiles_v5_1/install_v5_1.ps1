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
$BackupRoot = Join-Path $ProjectRoot "backups\player_profiles_v5_1_$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

$Files = @(
    "app.py",
    "src\assets\build_player_registry.py",
    "src\evaluation\registry_quality_audit.py"
)

foreach ($RelativePath in $Files) {
    $Source = Join-Path $SourceRoot $RelativePath
    $Target = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $Source)) {
        throw "Patch file is missing: $RelativePath"
    }
    if (Test-Path $Target) {
        $Backup = Join-Path $BackupRoot $RelativePath
        New-Item -ItemType Directory -Force -Path (Split-Path $Backup) | Out-Null
        Copy-Item $Target $Backup -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
    Copy-Item $Source $Target -Force
}

$PatchPages = Get-ChildItem -Path (Join-Path $SourceRoot "pages") -Filter "*.py" -File
foreach ($PatchPage in $PatchPages) {
    $TargetPage = Join-Path (Join-Path $ProjectRoot "pages") $PatchPage.Name
    if (Test-Path $TargetPage) {
        $BackupPage = Join-Path (Join-Path $BackupRoot "pages") $PatchPage.Name
        New-Item -ItemType Directory -Force -Path (Split-Path $BackupPage) | Out-Null
        Copy-Item $TargetPage $BackupPage -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $TargetPage) | Out-Null
    Copy-Item $PatchPage.FullName $TargetPage -Force
}

$ManualDirectory = Join-Path $ProjectRoot "data\manual"
$ManualFile = Join-Path $ManualDirectory "player_profile_overrides.csv"
if (-not (Test-Path $ManualFile)) {
    New-Item -ItemType Directory -Force -Path $ManualDirectory | Out-Null
    Copy-Item (
        Join-Path $SourceRoot "templates\player_profile_overrides_template.csv"
    ) $ManualFile -Force
}

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
    foreach ($PatchPage in $PatchPages) {
        $TargetPage = Join-Path (Join-Path $ProjectRoot "pages") $PatchPage.Name
        & python -m py_compile $TargetPage
        if ($LASTEXITCODE -ne 0) {
            throw "Python compilation failed: $($PatchPage.Name)"
        }
    }

    & python "src\assets\build_player_registry.py" --skip-player-history
    if ($LASTEXITCODE -ne 0) {
        throw "Player registry rebuild failed. The backup is at: $BackupRoot"
    }

    & python "src\evaluation\registry_quality_audit.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Player registry quality audit failed."
    }

    if (Test-Path "src\pipeline\run_operational_checks.py") {
        & python "src\pipeline\run_operational_checks.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Operational checks failed."
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
Write-Host "EPL Predictor v5.1 player profile upgrade installed." -ForegroundColor Green
Write-Host "Backup: $BackupRoot"
Write-Host "Manual verified overrides: $ManualFile"
Write-Host "Start the app with: python -m streamlit run app.py"
