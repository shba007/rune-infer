param(
    [Parameter(Mandatory = $true)]
    [string[]] $Images,
    [switch] $GpuLinearConv,
    [switch] $GpuNhwcBoundaries
)
$ErrorActionPreference = 'Stop'
# The container installs the pinned environment with uv and executes via uv run.
$taskRepository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskInputs = @($Images | ForEach-Object {
    $taskImage = (Resolve-Path -LiteralPath $_).Path
    if (-not (Test-Path -LiteralPath $taskImage -PathType Leaf)) {
        throw "Not an image file: $taskImage"
    }
    $taskImage
})
$taskWork = Join-Path $taskRepository ('target/depth-large-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskWork | Out-Null
$taskArguments = @('--output-dir', '/work/output')
if ($GpuLinearConv) { $taskArguments += '--gpu-linear-conv' }
if ($GpuNhwcBoundaries) { $taskArguments += '--gpu-nhwc-boundaries' }
for ($taskIndex = 0; $taskIndex -lt $taskInputs.Count; $taskIndex++) {
    $taskName = 'image-' + $taskIndex + [IO.Path]::GetExtension($taskInputs[$taskIndex])
    Copy-Item -LiteralPath $taskInputs[$taskIndex] -Destination (Join-Path $taskWork $taskName)
    $taskArguments += @('--image', "/work/$taskName")
}
docker build -t rune-infer-depth-converter:local $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Converter image build failed.' }
docker run --rm --memory=12g --memory-swap=16g --cpus=4 --cap-drop=ALL `
    --security-opt=no-new-privileges --mount "type=bind,source=$taskWork,target=/work" `
    rune-infer-depth-converter:local @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Conversion failed. Retained artifacts: $taskWork" }
Write-Output "Candidate and validation report: $taskWork/output"
Write-Output 'Windows native validation is still required before registering this artifact.'
