param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
& powershell -ExecutionPolicy Bypass -File (
    Join-Path $PSScriptRoot "install_v5_1.ps1"
) -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
