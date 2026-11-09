param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

chcp 65001 | Out-Null

$utf8Encoding = (
    [System.Text.UTF8Encoding]::new($false)
)

[Console]::InputEncoding = $utf8Encoding
[Console]::OutputEncoding = $utf8Encoding
$OutputEncoding = $utf8Encoding

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = Split-Path -Parent $PSScriptRoot

$pythonExecutable = Join-Path `
    $projectRoot `
    ".venv\Scripts\python.exe"

$lineupScript = Join-Path `
    $projectRoot `
    "src\context\fetch_lineups.py"

$reportScript = Join-Path `
    $projectRoot `
    "src\context\build_prematch_report.py"

$logDirectory = Join-Path `
    $projectRoot `
    "logs\lineups"

$lockFile = Join-Path `
    $projectRoot `
    "data\lineup_update.lock"

if (-not (Test-Path $pythonExecutable)) {
    throw "找不到虚拟环境 Python：$pythonExecutable"
}

if (-not (Test-Path $lineupScript)) {
    throw "找不到首发更新脚本：$lineupScript"
}

if (-not (Test-Path $reportScript)) {
    throw "找不到赛前报告脚本：$reportScript"
}

New-Item `
    -ItemType Directory `
    -Path $logDirectory `
    -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMddTHHmmss"
$logFile = Join-Path `
    $logDirectory `
    "lineup_update_$timestamp.log"

$lockStream = $null

try {
    try {
        $lockStream = [System.IO.File]::Open(
            $lockFile,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch [System.IO.IOException] {
        Write-Host "另一条首发更新任务正在运行，本次跳过。"
        exit 0
    }

    Set-Location $projectRoot

    $lineupArguments = @($lineupScript)

    if ($Force) {
        $lineupArguments += "--force"
    }

    Write-Host "开始检查官方首发阵容……"

    & $pythonExecutable @lineupArguments `
        2>&1 |
        Tee-Object -FilePath $logFile

    if ($LASTEXITCODE -ne 0) {
        throw "首发阵容更新失败。"
    }

    & $pythonExecutable $reportScript `
        2>&1 |
        Tee-Object -FilePath $logFile -Append

    if ($LASTEXITCODE -ne 0) {
        throw "统一赛前报告刷新失败。"
    }

    Write-Host "首发阵容和页面数据更新完成。"
}
finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }

    Remove-Item `
        $lockFile `
        -Force `
        -ErrorAction SilentlyContinue
}
