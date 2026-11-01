[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceTree,
    [string]$BuildDirectory = 'target/audiocpp-patched',
    [ValidateSet('cpu', 'cuda')][string]$Backend = 'cuda',
    [string]$CudaArchitectures = 'native',
    [string]$PrebuiltBackendDirectory = '',
    [switch]$AllowUnsupportedCudaCompiler,
    [int]$Jobs = 6
)

# Requires an existing checkout and CMake/compiler environment. Downloads no models
# or source, and never installs the result over a managed engine automatically.
$ErrorActionPreference = 'Stop'
$revision = '41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48'
$actual = git -C $SourceTree rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actual -ne $revision) { throw "Expected audio.cpp $revision" }
$patch = Join-Path $PSScriptRoot 'gguf-magic.patch'
git -C $SourceTree apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $SourceTree apply --check $patch
    if ($LASTEXITCODE -ne 0) { throw 'Source conflicts with the GGUF patch; no changes applied' }
    git -C $SourceTree apply $patch
    if ($LASTEXITCODE -ne 0) { throw 'Could not apply the GGUF patch' }
}
$cuda = if ($Backend -eq 'cuda') { 'ON' } else { 'OFF' }
$cudaFlags = @(if ($AllowUnsupportedCudaCompiler) { '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler' })
$reuse = 'OFF'
if ($PrebuiltBackendDirectory) {
    $version = & (Join-Path $PrebuiltBackendDirectory 'audiocpp_cli.exe') --version
    $matched = ($version -join "`n") -match 'git:\s+([a-f0-9]{7,40})'
    if ($LASTEXITCODE -ne 0 -or !$matched -or !$revision.StartsWith($matches[1])) {
        throw 'Prebuilt backends must come from the pinned Windows audio.cpp release'
    }
    $reuse = 'ON'
}
cmake -S $SourceTree -B $BuildDirectory -G Ninja -DCMAKE_BUILD_TYPE=Release `
    '-DAUDIOCPP_VERSION:STRING=0.9.3 (rune-gguf1)' -DAUDIOCPP_DEPLOYMENT_BUILD=ON `
    "-DENGINE_ENABLE_CUDA=$cuda" -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON `
    -DENGINE_BUILD_TESTS=ON "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
    "-DAUDIOCPP_PREBUILT_BACKENDS=$reuse" @cudaFlags
if ($LASTEXITCODE -ne 0) { throw 'audio.cpp configuration failed' }
cmake --build $BuildDirectory --target audiocpp_cli gguf_tensor_source_test --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'audio.cpp build failed' }
if ($PrebuiltBackendDirectory) {
    Get-ChildItem -LiteralPath $PrebuiltBackendDirectory -Filter '*.dll' -File |
        Copy-Item -Destination (Join-Path $BuildDirectory 'bin')
}
ctest --test-dir $BuildDirectory -R '^gguf_tensor_source_test$' --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'GGUF format regression failed' }
