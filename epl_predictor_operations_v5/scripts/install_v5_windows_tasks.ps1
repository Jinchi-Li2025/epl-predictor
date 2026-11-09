param(
    [string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent),
    [string]$DailyAt = "06:30",
    [int]$MatchdayIntervalMinutes = 15
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$DailyScript = Join-Path $ProjectRoot "scripts\run_daily_automation.ps1"
$MatchdayScript = Join-Path $ProjectRoot "scripts\run_matchday_automation.ps1"

if (-not (Test-Path $DailyScript) -or -not (Test-Path $MatchdayScript)) {
    throw "Install the v5 package before registering tasks."
}

$DailyAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$DailyScript`" -ProjectRoot `"$ProjectRoot`""
$DailyTrigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$MatchdayAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$MatchdayScript`" -ProjectRoot `"$ProjectRoot`""
$MatchdayTrigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(2)) -RepetitionInterval (New-TimeSpan -Minutes $MatchdayIntervalMinutes) -RepetitionDuration (New-TimeSpan -Days 3650)
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "EPL Predictor Daily Update" -Action $DailyAction -Trigger $DailyTrigger -Settings $Settings -Description "Daily EPL data, prediction and reliability update" -Force | Out-Null
Register-ScheduledTask -TaskName "EPL Predictor Matchday Refresh" -Action $MatchdayAction -Trigger $MatchdayTrigger -Settings $Settings -Description "Refresh near-kickoff context only when a fixture is due" -Force | Out-Null

Write-Host "Windows tasks installed." -ForegroundColor Green
Write-Host "Daily update: $DailyAt"
Write-Host "Matchday readiness check: every $MatchdayIntervalMinutes minutes"
Write-Host "The matchday task exits without remote calls when no fixture is near kickoff."
