param(
    [string]$ProjectRoot = (Get-Location).Path
)

$CoreInstaller = Get-ChildItem -Path $PSScriptRoot -Filter "*.ps1" -File |
    Where-Object { $_.Name -ne "install_upgrade.ps1" } |
    Sort-Object Length -Descending |
    Select-Object -First 1
if (-not $CoreInstaller) {
    throw "Core installer was not found. Extract the ZIP again."
}
& powershell -ExecutionPolicy Bypass -File $CoreInstaller.FullName -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
