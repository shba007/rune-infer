# demos/tts/driver.ps1
# Automates Text-to-Speech (TTS) demo request for VHS screencast recording

# Import shared helper utilities
$helpersPath = Join-Path $PSScriptRoot "..\common\driver_helpers.ps1"
if (Test-Path $helpersPath) {
    . $helpersPath
}

# 1. Wait for dashboard / server to be fully responsive
$serverUrl = "http://127.0.0.1:8081/v1/audio/speech"
$healthUrl = "http://127.0.0.1:8081/health"

if (Get-Command Wait-RuneServer -ErrorAction SilentlyContinue) {
    $null = Wait-RuneServer -Url $healthUrl -TimeoutSeconds 15
} else {
    Start-Sleep -Seconds 2
}

$headers = @{
    "Authorization" = "Bearer rune-demo-secure-key"
}

# 2. Text-to-Speech Request: Synthesize speech via Audio8 TTS Preview 0.6B
$ttsBody = @{
    model = "audio8-tts-preview-0.6b"
    input = "Welcome to Rune Infer. High-performance, local text-to-speech synthesis is operational."
    voice = "alloy"
    response_format = "wav"
} | ConvertTo-Json

$outputWav = "static/artifacts/audios/tts_demo.wav"

try {
    # Ensure destination directory exists
    $outDir = Split-Path -Parent $outputWav
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }

    # Send synthesis request and write audio payload to disk
    $response = Invoke-WebRequest -Uri $serverUrl -Method Post -Headers $headers -ContentType "application/json" -Body $ttsBody -OutFile $outputWav -TimeoutSec 30
} catch {
    # Handled gracefully if server is shutting down
}

# 3. Hold view on dashboard to observe live inference metrics, request log, and tok/s rate
Start-Sleep -Seconds 6
