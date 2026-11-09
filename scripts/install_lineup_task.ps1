param(
    [string]$TaskName = "EPL Predictor - Lineup Refresh"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path `
    $projectRoot `
    "scripts\run_lineup_update.ps1"

if (-not (Test-Path $runner)) {
    throw "找不到首发更新入口：$runner"
}

$actionArguments = (
    '-NoProfile -ExecutionPolicy Bypass -File "' +
    $runner +
    '"'
)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $actionArguments `
    -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description (
        "每15分钟检查临近开赛的英超官方首发；" +
        "非临场比赛不会调用阵容API。"
    ) `
    -Force | Out-Null

Write-Host "计划任务已安装：$TaskName"
Write-Host "检查频率：每15分钟"
Write-Host "脚本只会在开赛前75分钟至开赛后30分钟调用阵容API。"
Write-Host "可在任务计划程序中查看或停用该任务。"
