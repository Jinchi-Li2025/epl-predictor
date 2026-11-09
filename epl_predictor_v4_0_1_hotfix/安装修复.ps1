param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$Installer = Join-Path $PSScriptRoot "install_hotfix.ps1"
& powershell -ExecutionPolicy Bypass -File $Installer -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
