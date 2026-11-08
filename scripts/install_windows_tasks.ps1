$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

$runnerScript = Join-Path `
    $projectRoot `
    "scripts\run_production_update.ps1"

if (-not (Test-Path $runnerScript)) {
    throw "找不到自动运行脚本：$runnerScript"
}

$currentUser = (
    [System.Security.Principal.WindowsIdentity]::GetCurrent()
).Name

$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (
        New-TimeSpan -Hours 2
    ) `
    -MultipleInstances IgnoreNew

$dailyActionArguments = (
    '-NoProfile ' +
    '-ExecutionPolicy Bypass ' +
    '-File "' +
    $runnerScript +
    '"'
)

$dailyAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $dailyActionArguments `
    -WorkingDirectory $projectRoot

$dailyTriggers = @(
    New-ScheduledTaskTrigger `
        -Daily `
        -At "06:15"

    New-ScheduledTaskTrigger `
        -Daily `
        -At "12:15"

    New-ScheduledTaskTrigger `
        -Daily `
        -At "18:15"

    New-ScheduledTaskTrigger `
        -Daily `
        -At "23:45"
)

$dailyTask = New-ScheduledTask `
    -Action $dailyAction `
    -Trigger $dailyTriggers `
    -Principal $principal `
    -Settings $settings `
    -Description (
        "自动同步英超结果、生成下一轮预测、更新赛前上下文和归档历史预测。"
    )

Register-ScheduledTask `
    -TaskName "EPL Predictor - Production Update" `
    -InputObject $dailyTask `
    -Force | Out-Null

$validationActionArguments = (
    '-NoProfile ' +
    '-ExecutionPolicy Bypass ' +
    '-File "' +
    $runnerScript +
    '" ' +
    '-WithValidation'
)

$validationAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $validationActionArguments `
    -WorkingDirectory $projectRoot

$validationTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Sunday `
    -At "07:00"

$validationTask = New-ScheduledTask `
    -Action $validationAction `
    -Trigger $validationTrigger `
    -Principal $principal `
    -Settings $settings `
    -Description (
        "每周运行生产更新并重新计算当前赛季外部验证指标。"
    )

Register-ScheduledTask `
    -TaskName "EPL Predictor - Weekly Validation" `
    -InputObject $validationTask `
    -Force | Out-Null

Write-Host ""
Write-Host "Windows计划任务安装完成。"
Write-Host ""

Get-ScheduledTask `
    -TaskName "EPL Predictor*" |
    Select-Object `
        TaskName,
        State |
    Format-Table -AutoSize