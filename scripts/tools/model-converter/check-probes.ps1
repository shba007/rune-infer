param(
    [Parameter(Mandatory = $true)] [string] $ProbeDirectory,
    [Parameter(Mandatory = $true)] [string] $Worker,
    [Parameter(Mandatory = $true)] [string] $Checker,
    [Parameter(Mandatory = $true)] [string] $Runtime,
    [switch] $IrGraph
)
$ErrorActionPreference = 'Stop'
$taskDirectory = (Resolve-Path -LiteralPath $ProbeDirectory).Path
$taskWorker = (Resolve-Path -LiteralPath $Worker).Path
$taskChecker = (Resolve-Path -LiteralPath $Checker).Path
$taskRuntime = (Resolve-Path -LiteralPath $Runtime).Path
$taskProof = Get-Content -LiteralPath (Join-Path $taskDirectory 'probes.json') -Raw | ConvertFrom-Json
if ($taskProof.source_sha256 -cne '6f03564f13abfae18185581a809cb5c0ee5c0a425dc5f85102194b2b321eba3d' -or
    $taskProof.probes.Count -lt 1 -or $taskProof.probes.Count -gt 32) { throw 'Invalid diagnostic manifest.' }
$taskNames = @('RUNE_LITERT_CANDIDATE_SHA256', 'RUNE_LITERT_PROBE_SHAPE', 'RUNE_LITERT_VALIDATION_DIAGNOSTICS', 'RUNE_LITERT_VALIDATE_NHWC', 'RUNE_LITERT_VALIDATE_IR')
$taskPrevious = @{}
foreach ($taskName in $taskNames) { $taskPrevious[$taskName] = [Environment]::GetEnvironmentVariable($taskName) }
$taskOutput = Join-Path $taskDirectory ('windows-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskOutput | Out-Null
try {
    $env:RUNE_LITERT_VALIDATION_DIAGNOSTICS = '1'
    $env:RUNE_LITERT_VALIDATE_NHWC = $null
    $env:RUNE_LITERT_VALIDATE_IR = if ($IrGraph) { '1' } else { $null }
    $taskResults = foreach ($taskProbe in $taskProof.probes) {
        if ($taskProbe.operator -notmatch '^\d+$' -or $taskProbe.model_sha256 -cnotmatch '^[a-f0-9]{64}$') { throw 'Invalid probe identity.' }
        $taskPrefix = Join-Path $taskDirectory "op-$($taskProbe.operator)"
        if ((Get-FileHash -LiteralPath "$taskPrefix.reference.f32" -Algorithm SHA256).Hash.ToLowerInvariant() -cne $taskProbe.reference_sha256 -or
            (Get-FileHash -LiteralPath "$taskPrefix.input.f32" -Algorithm SHA256).Hash.ToLowerInvariant() -cne $taskProof.input_sha256) { throw 'Probe fixture checksum mismatch.' }
        $env:RUNE_LITERT_CANDIDATE_SHA256 = $taskProbe.model_sha256
        $env:RUNE_LITERT_PROBE_SHAPE = $taskProbe.shape -join ','
        $taskLog = Join-Path $taskOutput "op-$($taskProbe.operator).log"
        & $taskChecker $taskWorker $taskRuntime "$taskPrefix.tflite" $taskPrefix *> $taskLog
        $taskExit = $LASTEXITCODE
        $taskText = Get-Content -LiteralPath $taskLog -Raw
        $taskMatch = [regex]::Match($taskText, 'max_absolute_error=([\d.Ee+-]+).*failed_values=(\d+)')
        [ordered]@{operator=$taskProbe.operator; kind=$taskProbe.kind; shape=$taskProbe.shape; passed=($taskExit -eq 0);
            max_absolute_error=if ($taskMatch.Success) { [double]::Parse($taskMatch.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) } else { $null };
            failed_values=if ($taskMatch.Success) { [int]$taskMatch.Groups[2].Value } else { $null }; exit_code=$taskExit; log=$taskLog}
    }
    [ordered]@{source_sha256=$taskProof.source_sha256; worker_sha256=(Get-FileHash -LiteralPath $taskWorker -Algorithm SHA256).Hash.ToLowerInvariant();
        graph=if ($IrGraph) { 'ir' } else { 'legacy' }; tolerance=@{atol=0.001;rtol=0.001}; probes=@($taskResults)} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskOutput 'report.json') -Encoding utf8
    $taskResults | ForEach-Object { Write-Output "operator=$($_.operator) kind=$($_.kind) passed=$($_.passed) max_error=$($_.max_absolute_error) failures=$($_.failed_values)" }
    Write-Output "Report: $taskOutput/report.json"
    if (@($taskResults | Where-Object { -not $_.passed }).Count) { throw 'One or more diagnostic checkpoints failed; inspect the report.' }
} finally {
    foreach ($taskName in $taskNames) { [Environment]::SetEnvironmentVariable($taskName, $taskPrevious[$taskName]) }
}
