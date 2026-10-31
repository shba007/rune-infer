param(
    [Parameter(Mandatory = $true)]
    [string] $Report,
    [string] $Runtime
)
$ErrorActionPreference = 'Stop'
$taskRepository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskReportPath = (Resolve-Path -LiteralPath $Report).Path
$taskProof = Get-Content -LiteralPath $taskReportPath -Raw | ConvertFrom-Json
if ($taskProof.status -ne 'linux_cpu_parity_passed' -or $taskProof.checks.Count -lt 1 -or
    $taskProof.checks.Count -gt 32 -or @($taskProof.checks | Where-Object { -not $_.passed }).Count) {
    throw 'A completed Linux CPU parity report is required.'
}
if ([IO.Path]::GetFileName($taskProof.candidate.file) -ne $taskProof.candidate.file) {
    throw 'The candidate must be a file beside the report.'
}
$taskDirectory = Split-Path -Parent $taskReportPath
$taskCandidate = (Resolve-Path -LiteralPath (Join-Path $taskDirectory $taskProof.candidate.file)).Path
if (-not $Runtime) { $Runtime = Join-Path $taskRepository 'bin/litert/libLiteRt.dll' }
$taskRuntime = (Resolve-Path -LiteralPath $Runtime).Path
$taskPreviousHash = $env:RUNE_LITERT_CANDIDATE_SHA256
Push-Location $taskRepository
try {
    $taskBuild = @(cargo build --locked --lib --target-dir target/ci-check --message-format=json)
    if ($LASTEXITCODE -ne 0) { throw 'Infer library build failed.' }
    $taskArtifacts = @($taskBuild | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.reason -eq 'compiler-artifact' })
    $taskAnyhow = @($taskArtifacts | Where-Object { $_.target.name -eq 'anyhow' } |
        ForEach-Object { $_.filenames } | Where-Object { $_.EndsWith('.rlib') })
    $taskSha2 = @($taskArtifacts | Where-Object { $_.target.name -eq 'sha2' } |
        ForEach-Object { $_.filenames } | Where-Object { $_.EndsWith('.rlib') })
    if ($taskAnyhow.Count -ne 1 -or $taskSha2.Count -ne 1) { throw 'Ambiguous validation dependencies.' }
    rustc --edition 2024 -D warnings "$PSScriptRoot/candidate_worker.rs" `
        -L dependency=target/ci-check/debug/deps --extern "sha2=$($taskSha2[0])" `
        --extern "anyhow=$($taskAnyhow[0])" -o target/depth-large-candidate-worker.exe
    if ($LASTEXITCODE -ne 0) { throw 'Candidate worker build failed.' }
    rustc --edition 2024 -D warnings "$PSScriptRoot/native_check.rs" -o target/depth-large-native-check.exe
    if ($LASTEXITCODE -ne 0) { throw 'Native checker build failed.' }
    $taskWorker = (Resolve-Path target/depth-large-candidate-worker.exe).Path
    $env:RUNE_LITERT_CANDIDATE_SHA256 = $taskProof.candidate.sha256
    for ($taskIndex = 0; $taskIndex -lt $taskProof.checks.Count; $taskIndex++) {
        $taskPrefix = Join-Path $taskDirectory "fixture-$taskIndex"
        & target/depth-large-native-check.exe $taskWorker $taskRuntime $taskCandidate $taskPrefix
        if ($LASTEXITCODE -ne 0) { throw "Windows CPU parity failed for fixture $taskIndex." }
    }
} finally {
    $env:RUNE_LITERT_CANDIDATE_SHA256 = $taskPreviousHash
    Pop-Location
}
