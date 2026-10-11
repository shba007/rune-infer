# Rune Infer installer for Windows (PowerShell 5.1+ / 7+).
#
#   irm https://raw.githubusercontent.com/shba007/rune-infer/main/install.ps1 | iex
#
# Environment overrides:
#   RUNE_VERSION      release tag to install (default: latest), e.g. v0.2.0
#   RUNE_INSTALL_DIR  where to put the binary (default: %LOCALAPPDATA%\Programs\rune-infer)
#   RUNE_REPO         GitHub repo (default: shba007/rune-infer)
#   RUNE_BASE_URL     download base URL or local directory (mirrors / offline installs)
#   RUNE_SKIP_VERIFY  set to 1 to skip the SHA-256 check (not recommended)
#   RUNE_NO_MODIFY_PATH  set to 1 to leave your PATH untouched

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is ~10x faster without the progress bar
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

function Get-Setting($name, $default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($v)) { $default } else { $v }
}

$repo = Get-Setting 'RUNE_REPO' 'shba007/rune-infer'
$version = Get-Setting 'RUNE_VERSION' 'latest'
$installDir = Get-Setting 'RUNE_INSTALL_DIR' (Join-Path $env:LOCALAPPDATA 'Programs\rune-infer')
$bin = 'rune-infer'

$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
switch ($arch) {
    'X64'   { }
    'Arm64' { Write-Warning 'No native Arm64 build yet; installing the x64 build (runs under emulation).' }
    default { throw "Unsupported CPU architecture '$arch'. Build from source: cargo install --git https://github.com/$repo" }
}
$archive = "$bin-windows-x64.zip"

$base = Get-Setting 'RUNE_BASE_URL' ''
if ($base) { $base = $base.TrimEnd('/', '\') }
elseif ($version -eq 'latest') { $base = "https://github.com/$repo/releases/latest/download" }
else {
    if (-not $version.StartsWith('v')) { $version = "v$version" }
    $base = "https://github.com/$repo/releases/download/$version"
}

function Fetch($name, $dest) {
    if (Test-Path -LiteralPath $base -PathType Container) {
        $src = Join-Path $base $name
        if (-not (Test-Path -LiteralPath $src)) { throw "$src not found" }
        Copy-Item -LiteralPath $src -Destination $dest
    } else {
        try { Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile $dest }
        catch { throw "Download failed: $base/$name`n$($_.Exception.Message)" }
    }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rune-infer-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "Installing Rune Infer (windows-x64, $version)..."
    $zip = Join-Path $tmp $archive
    Fetch $archive $zip

    if ((Get-Setting 'RUNE_SKIP_VERIFY' '0') -eq '1') {
        Write-Warning 'Skipping checksum verification.'
    } else {
        $sumFile = "$zip.sha256"
        Fetch "$archive.sha256" $sumFile
        $expected = ((Get-Content -LiteralPath $sumFile -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
        $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected -ne $actual) { throw "Checksum mismatch for $archive (expected $expected, got $actual). Aborting." }
        Write-Host 'Checksum verified.'
    }

    $extract = Join-Path $tmp 'x'
    Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
    $exe = Join-Path $extract "$bin.exe"
    if (-not (Test-Path -LiteralPath $exe)) { throw "Archive did not contain $bin.exe" }

    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    $target = Join-Path $installDir "$bin.exe"
    $upgrade = Test-Path -LiteralPath $target
    try { Copy-Item -LiteralPath $exe -Destination $target -Force }
    catch { throw "Cannot replace $target - is Rune Infer still running? Close it and run the installer again." }

    if ($upgrade) { Write-Host "Upgraded: $target" } else { Write-Host "Installed: $target" }
    try { & $target --version } catch { Write-Warning "Installed binary did not start: $($_.Exception.Message)" }

    # Add to the *user* PATH (no admin needed) and to this session.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    if ((Get-Setting 'RUNE_NO_MODIFY_PATH' '0') -eq '1') {
        Write-Host "Skipped PATH update. Add $installDir to your PATH manually."
    } elseif ($entries -notcontains $installDir) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $installDir) -join ';'), 'User')
        $env:Path = "$env:Path;$installDir"
        Write-Host "Added $installDir to your PATH (open a new terminal for other windows to see it)."
    }

    Write-Host ''
    Write-Host "Next: run 'rune-infer' - a short setup wizard creates your config in $env:USERPROFILE\.rune-infer."
    Write-Host 'Optional: install FFmpeg (winget install Gyan.FFmpeg) for video input.'
}
finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
