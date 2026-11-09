$ErrorActionPreference = "Stop"
foreach ($Name in @("EPL Predictor Daily Update", "EPL Predictor Matchday Refresh")) {
    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "Removed: $Name"
    }
}
