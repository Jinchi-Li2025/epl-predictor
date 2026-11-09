param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Push-Location $ProjectRoot
try {
    $Blockers = New-Object System.Collections.Generic.List[string]
    $Warnings = New-Object System.Collections.Generic.List[string]

    & git rev-parse --is-inside-work-tree *> $null
    if ($LASTEXITCODE -ne 0) {
        $Blockers.Add("The project is not a Git repository.")
    }

    $TrackedFiles = @(& git ls-files)
    $SensitivePatterns = @(
        '(^|/)\.env$',
        '(^|/)\.streamlit/secrets\.toml$',
        '\.(pem|key)$'
    )
    foreach ($File in $TrackedFiles) {
        foreach ($Pattern in $SensitivePatterns) {
            if ($File -match $Pattern) {
                $Blockers.Add("Sensitive file is tracked: $File")
                break
            }
        }
    }

    $EnvironmentHistory = @(& git log --all --format=%H -- .env)
    if ($EnvironmentHistory.Count -gt 0) {
        $Blockers.Add(
            ".env appears in Git history. Rotate exposed keys and clean history before a public push."
        )
    }

    $UnsafeLocalFiles = @(
        $TrackedFiles | Where-Object { $_ -match '^(\.venv|venv|backups)/' }
    )
    if ($UnsafeLocalFiles.Count -gt 0) {
        $Blockers.Add(
            "$($UnsafeLocalFiles.Count) local environment/backup files are tracked."
        )
    }

    foreach ($File in $TrackedFiles) {
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) {
            continue
        }
        $Size = (Get-Item -LiteralPath $File).Length
        if ($Size -ge 100MB) {
            $Blockers.Add("GitHub rejects files of 100 MiB or larger: $File")
        } elseif ($Size -ge 50MB) {
            $Warnings.Add("Large tracked file should be reviewed: $File")
        }
    }

    $SecretAssignment = '(DEEPSEEK_API_KEY|FOOTBALL_DATA_API_KEY|FOOTBALL_DATA_TOKEN|API_FOOTBALL_KEY)\s*=\s*["'']?[^\s"'']{8,}'
    foreach ($File in $TrackedFiles) {
        if ($File -eq ".env.example" -or -not (Test-Path -LiteralPath $File -PathType Leaf)) {
            continue
        }
        try {
            if (Select-String -LiteralPath $File -Pattern $SecretAssignment -Quiet) {
                $Blockers.Add("Possible hard-coded API key in: $File")
            }
        } catch {
            # Binary or unsupported files are covered by file and extension checks.
        }
    }

    $PythonFiles = @($TrackedFiles | Where-Object { $_.EndsWith(".py") })
    foreach ($File in $PythonFiles) {
        & python -m py_compile $File
        if ($LASTEXITCODE -ne 0) {
            $Blockers.Add("Python compilation failed: $File")
        }
    }

    $TrackedMedia = @(
        $TrackedFiles | Where-Object {
            $_ -match '^assets/(stadiums|teams/crests|teams/kits)/'
        }
    )
    if ($TrackedMedia.Count -gt 0) {
        $Warnings.Add(
            "$($TrackedMedia.Count) downloaded media files are tracked; review licences before a public release."
        )
    }
    $GeneratedTracked = @(
        $TrackedFiles | Where-Object {
            $_ -match '^(data/raw|data/processed|data/context|reports|logs|artifacts)/'
        }
    )
    if ($GeneratedTracked.Count -gt 0) {
        $Warnings.Add(
            "$($GeneratedTracked.Count) generated data/report files are tracked; review them before a public release."
        )
    }

    Write-Host ""
    Write-Host "GitHub preflight completed."
    Write-Host "Tracked files: $($TrackedFiles.Count)"
    Write-Host "Python files compiled: $($PythonFiles.Count)"
    Write-Host "Blockers: $($Blockers.Count); warnings: $($Warnings.Count)"
    foreach ($Item in $Warnings) {
        Write-Host "WARNING: $Item" -ForegroundColor Yellow
    }
    foreach ($Item in $Blockers) {
        Write-Host "BLOCKER: $Item" -ForegroundColor Red
    }
    Write-Host "No secret values were printed."

    if ($Blockers.Count -gt 0) {
        exit 1
    }
    Write-Host "Repository is ready for a private first push." -ForegroundColor Green
} finally {
    Pop-Location
}
