# demos/record.ps1
# Master orchestrator for Rune Infer VHS screencast recordings

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Name = "all",

    [switch]$Build,
    [switch]$ValidateOnly,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

# Ensure we operate from repository root
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "=== Rune Infer Demo Recording Orchestrator ===" -ForegroundColor Cyan
Write-Host "Repository root: $repoRoot" -ForegroundColor Gray

# 1. Check for VHS executable
$vhsCmd = Get-Command "vhs" -ErrorAction SilentlyContinue
if (-not $vhsCmd) {
    Write-Error "Charm VHS CLI ('vhs') was not found on PATH.`nPlease install it: https://github.com/charmbracelet/vhs"
    exit 1
}

# 2. Discover available demo tapes (excluding template/common)
$tapeFiles = Get-ChildItem -Path (Join-Path $PSScriptRoot "*/*.tape") -File | Where-Object {
    $_.Directory.Name -ne "common" -and $_.Name -notlike "*template*"
}
if ($tapeFiles.Count -eq 0) {
    Write-Warning "No active .tape files found in $PSScriptRoot/*/"
    exit 0
}

# Filter by requested demo name
if ($Name -ne "all") {
    $matched = $tapeFiles | Where-Object {
        $_.Directory.Name -like "*$Name*" -or $_.BaseName -like "*$Name*"
    }
    if (-not $matched) {
        Write-Error "No demo tape matching '$Name'. Available demos: $(($tapeFiles | ForEach-Object { $_.Directory.Name }) -join ', ')"
        exit 1
    }
    $tapeFiles = $matched
}

Write-Host "Selected demos: $(($tapeFiles | ForEach-Object { $_.Directory.Name }) -join ', ')" -ForegroundColor Yellow

# 3. Check or build release binary if not validate-only
if (-not $ValidateOnly) {
    $binName = if ($IsWindows -or $env:OS -like "*Windows*") { "rune-infer.exe" } else { "rune-infer" }
    $binPath = Join-Path $repoRoot "target/release/$binName"

    if ($Build -or (-not (Test-Path $binPath))) {
        Write-Host "`nBuilding release binary (target/release/$binName)..." -ForegroundColor Cyan
        & cargo build --release --bin rune-infer
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Release build failed."
            exit $LASTEXITCODE
        }
    }
}

# 4. Process each selected demo tape
$results = @()

foreach ($tape in $tapeFiles) {
    $demoName = $tape.Directory.Name
    $relativeTape = $tape.FullName.Substring($repoRoot.Length).TrimStart('\', '/')

    Write-Host "`n--> [$demoName] Validating $relativeTape..." -ForegroundColor Cyan
    & vhs validate $relativeTape
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tape validation failed for $relativeTape."
        exit $LASTEXITCODE
    }

    if ($ValidateOnly) {
        Write-Host "    Validation passed." -ForegroundColor Green
        continue
    }

    Write-Host "--> [$demoName] Rendering recording with VHS..." -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $vhsArgs = @($relativeTape)
    if ($Quiet) { $vhsArgs += "-q" }

    & vhs @vhsArgs
    $sw.Stop()

    if ($LASTEXITCODE -eq 0) {
        # Determine output file from tape content
        $outputLine = Get-Content $tape.FullName | Where-Object { $_ -match '^\s*Output\s+(.+)$' } | Select-Object -First 1
        $outputFile = ""
        $fileSize = ""
        if ($outputLine -and $outputLine -match '^\s*Output\s+(.+)$') {
            $rawOutput = $Matches[1].Trim()
            $resolvedOutput = Join-Path $repoRoot $rawOutput
            if (Test-Path $resolvedOutput) {
                $outputFile = $rawOutput
                $len = (Get-Item $resolvedOutput).Length
                $fileSize = "{0:N2} MB" -f ($len / 1MB)
            }
        }

        Write-Host "    Completed in $($sw.Elapsed.ToString('mm\:ss')) -> $outputFile ($fileSize)" -ForegroundColor Green
        $results += [PSCustomObject]@{
            Demo = $demoName
            Status = "Success"
            Duration = $sw.Elapsed.ToString('mm\:ss')
            Output = $outputFile
            Size = $fileSize
        }
    } else {
        Write-Host "    Recording failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $results += [PSCustomObject]@{
            Demo = $demoName
            Status = "Failed"
            Duration = $sw.Elapsed.ToString('mm\:ss')
            Output = "N/A"
            Size = "N/A"
        }
    }
}

if (-not $ValidateOnly -and $results.Count -gt 0) {
    Write-Host "`n=== Recording Summary ===" -ForegroundColor Cyan
    $results | Format-Table -AutoSize
}
