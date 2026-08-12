[CmdletBinding()]
param(
    [string]$Name = "ntsb"
)

$currentDirectory = (Get-Location).Path.Replace('\', '/')
$commonGitDirectory = git -c "safe.directory=$currentDirectory" `
    rev-parse --path-format=absolute --git-common-dir 2>$null
if ($LASTEXITCODE -ne 0 -or -not $commonGitDirectory) {
    throw "Run this script from an AeroLLM Git worktree."
}

$repositoryRoot = Split-Path -Parent $commonGitDirectory.Trim()
$environmentFile = Join-Path $repositoryRoot ".secrets\$Name.env"
if (-not (Test-Path -LiteralPath $environmentFile -PathType Leaf)) {
    throw "Local environment file not found: $environmentFile"
}

$loaded = [System.Collections.Generic.List[string]]::new()
foreach ($line in Get-Content -LiteralPath $environmentFile) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
    }

    $separator = $trimmed.IndexOf("=")
    if ($separator -lt 1) {
        throw "Invalid environment entry in $environmentFile. Expected NAME=VALUE."
    }

    $variableName = $trimmed.Substring(0, $separator).Trim()
    $variableValue = $trimmed.Substring($separator + 1).Trim()
    if ($variableName -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
        throw "Invalid environment variable name in $environmentFile."
    }
    if (-not $variableValue) {
        throw "Environment variable $variableName has no value in $environmentFile."
    }
    if ($variableValue -match '^<.*>$') {
        throw "Replace the placeholder value for $variableName in $environmentFile."
    }

    [Environment]::SetEnvironmentVariable($variableName, $variableValue, "Process")
    $loaded.Add($variableName)
}

if ($loaded.Count -eq 0) {
    throw "No environment variables were found in $environmentFile."
}

Write-Host "Loaded $($loaded.Count) local variable(s): $($loaded -join ', ')"
Write-Host "Values were not printed. Variables apply to this PowerShell session."
