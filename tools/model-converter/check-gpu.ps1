param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $RuntimeDirectory,
    [Parameter(Mandatory = $true)] [string] $ShaderDirectory,
    [Parameter(Mandatory = $true)] [string] $FixturePrefix,
    [string] $ConversionReport,
    [switch] $IrGraph
)
$ErrorActionPreference = 'Stop'
$taskRepository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskModel = (Resolve-Path -LiteralPath $Model).Path
$taskRuntime = (Resolve-Path -LiteralPath (Join-Path $RuntimeDirectory 'libLiteRt.dll')).Path
$taskShaders = (Resolve-Path -LiteralPath $ShaderDirectory).Path
$taskInput = (Resolve-Path -LiteralPath "$FixturePrefix.input.f32").Path
$taskReference = (Resolve-Path -LiteralPath "$FixturePrefix.reference.f32").Path
$taskPrefix = $taskInput.Substring(0, $taskInput.Length - '.input.f32'.Length)
$taskExpectedHash = '6f03564f13abfae18185581a809cb5c0ee5c0a425dc5f85102194b2b321eba3d'
$taskNhwc = $false
$taskConversionReport = $null
if ($ConversionReport) {
    $taskConversionReport = (Resolve-Path -LiteralPath $ConversionReport).Path
    $taskProof = Get-Content -LiteralPath $taskConversionReport -Raw | ConvertFrom-Json
    if ($taskProof.status -ne 'linux_cpu_parity_passed' -or $taskProof.checks.Count -lt 1 -or
        @($taskProof.checks | Where-Object { -not $_.passed }).Count -or
        $taskProof.candidate.sha256 -cnotmatch '^[a-f0-9]{64}$') {
        throw 'A completed CPU parity report with a valid candidate SHA-256 is required.'
    }
    $taskExpectedHash = $taskProof.candidate.sha256
    $taskNhwc = $taskProof.input.layout -ceq 'NHWC'
}
$taskPreviousHash = $env:RUNE_LITERT_CANDIDATE_SHA256
$taskPreviousDiagnostics = $env:RUNE_LITERT_VALIDATION_DIAGNOSTICS
$taskPreviousIr = $env:RUNE_LITERT_VALIDATE_IR
$taskPreviousNhwc = $env:RUNE_LITERT_VALIDATE_NHWC
$taskPreviousHooks = $env:CARGO_HUSKY_DONT_INSTALL_HOOKS
Push-Location $taskRepository
try {
    # Deliberately no downloads, installation, catalog edits or Python dependency.
    # GPU worker verifies pinned runtime, WebGPU and DXC DLLs before loading them.
    $taskDirectory = Join-Path $taskRepository ('target/litert-gpu-check-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $taskDirectory | Out-Null
    foreach ($taskName in @('dxcompiler.dll', 'dxil.dll')) {
        Copy-Item -LiteralPath (Join-Path $taskShaders $taskName) -Destination $taskDirectory
    }
    $env:CARGO_HUSKY_DONT_INSTALL_HOOKS = '1'
    $taskBuild = @(cargo build --locked --lib --target-dir target/ci-check --message-format=json)
    if ($LASTEXITCODE -ne 0) { throw 'Infer library build failed.' }
    $taskArtifacts = @($taskBuild | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.reason -eq 'compiler-artifact' })
    $taskAnyhow = @($taskArtifacts | Where-Object { $_.target.name -eq 'anyhow' } |
        ForEach-Object { $_.filenames } | Where-Object { $_.EndsWith('.rlib') })
    $taskSha = @($taskArtifacts | Where-Object { $_.target.name -eq 'sha2' } |
        ForEach-Object { $_.filenames } | Where-Object { $_.EndsWith('.rlib') })
    if ($taskAnyhow.Count -ne 1 -or $taskSha.Count -ne 1) { throw 'Ambiguous validation dependencies.' }
    $taskWorker = Join-Path $taskDirectory 'gpu-worker.exe'
    $taskChecker = Join-Path $taskDirectory 'native-check.exe'
    rustc --edition 2024 -D warnings "$PSScriptRoot/gpu_worker.rs" `
        -L dependency=target/ci-check/debug/deps --extern "sha2=$($taskSha[0])" `
        --extern "anyhow=$($taskAnyhow[0])" -o $taskWorker
    if ($LASTEXITCODE -ne 0) { throw 'GPU validation worker build failed.' }
    rustc --edition 2024 -D warnings "$PSScriptRoot/native_check.rs" -o $taskChecker
    if ($LASTEXITCODE -ne 0) { throw 'Native checker build failed.' }
    $env:RUNE_LITERT_CANDIDATE_SHA256 = $taskExpectedHash
    $env:RUNE_LITERT_VALIDATION_DIAGNOSTICS = '1'
    $env:RUNE_LITERT_VALIDATE_IR = if ($IrGraph) { '1' } else { $null }
    $env:RUNE_LITERT_VALIDATE_NHWC = if ($taskNhwc) { '1' } else { $null }
    $taskLog = Join-Path $taskDirectory 'validation.log'
    & $taskChecker $taskWorker $taskRuntime $taskModel $taskPrefix *> $taskLog
    $taskExit = $LASTEXITCODE
    $taskReport = [ordered]@{
        status = if ($taskExit -eq 0) { 'gpu_fp32_parity_passed' } else { 'gpu_validation_failed' }
        backend = 'Windows WebGPU Direct3D12, GPU with CPU operations permitted'
        graph = if ($IrGraph) { 'ir' } else { 'legacy' }
        conversion_report = $taskConversionReport
        nhwc_boundaries = $taskNhwc
        tolerance = @{ atol = 0.001; rtol = 0.001 }
        model = @{ path = $taskModel; sha256 = (Get-FileHash -LiteralPath $taskModel -Algorithm SHA256).Hash.ToLowerInvariant() }
        input = @{ path = $taskInput; sha256 = (Get-FileHash -LiteralPath $taskInput -Algorithm SHA256).Hash.ToLowerInvariant() }
        reference = @{ path = $taskReference; sha256 = (Get-FileHash -LiteralPath $taskReference -Algorithm SHA256).Hash.ToLowerInvariant() }
        log = $taskLog
        exit_code = $taskExit
    }
    $taskReport | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskDirectory 'report.json') -Encoding utf8
    Write-Output "Validation report: $taskDirectory/report.json"
    if ($taskExit -ne 0) { throw "GPU validation failed; inspect $taskLog. Production configuration was not changed." }
} finally {
    $env:RUNE_LITERT_CANDIDATE_SHA256 = $taskPreviousHash
    $env:RUNE_LITERT_VALIDATION_DIAGNOSTICS = $taskPreviousDiagnostics
    $env:RUNE_LITERT_VALIDATE_IR = $taskPreviousIr
    $env:RUNE_LITERT_VALIDATE_NHWC = $taskPreviousNhwc
    $env:CARGO_HUSKY_DONT_INSTALL_HOOKS = $taskPreviousHooks
    Pop-Location
}
