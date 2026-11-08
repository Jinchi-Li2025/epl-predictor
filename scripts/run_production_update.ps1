param(
    [switch]$SkipFetch,
    [switch]$WithValidation
)

$ErrorActionPreference = "Stop"

# 统一使用UTF-8，支持球员姓名中的欧洲特殊字符。
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

$pipelineScript = Join-Path `
    $projectRoot `
    "src\pipeline\run_daily_update.py"

$scheduledLogDirectory = Join-Path `
    $projectRoot `
    "logs\scheduled"

$lockFile = Join-Path `
    $projectRoot `
    "data\pipeline_update.lock"

if (-not (Test-Path $pythonExecutable)) {
    throw "找不到虚拟环境 Python：$pythonExecutable"
}

if (-not (Test-Path $pipelineScript)) {
    throw "找不到生产流水线：$pipelineScript"
}

New-Item `
    -ItemType Directory `
    -Path $scheduledLogDirectory `
    -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMddTHHmmss"

$scheduledLogFile = Join-Path `
    $scheduledLogDirectory `
    "scheduled_update_$timestamp.log"

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
        Write-Host "已有另一条更新流水线正在运行，本次任务跳过。"
        exit 0
    }

    Set-Location $projectRoot

    $pipelineArguments = @(
        $pipelineScript
    )

    if ($SkipFetch) {
        $pipelineArguments += "--skip-fetch"
    }

    if ($WithValidation) {
        $pipelineArguments += "--with-validation"
    }

    Write-Host "项目目录：$projectRoot"
    Write-Host "Python：$pythonExecutable"
    Write-Host "计划任务日志：$scheduledLogFile"
    Write-Host "开始运行生产更新流水线……"

    & $pythonExecutable @pipelineArguments `
        2>&1 |
        Tee-Object -FilePath $scheduledLogFile

    $pipelineExitCode = $LASTEXITCODE

    if ($pipelineExitCode -ne 0) {
        throw (
            "生产更新流水线失败，退出代码：" +
            $pipelineExitCode
        )
    }

    Write-Host "生产更新流水线运行成功。"
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