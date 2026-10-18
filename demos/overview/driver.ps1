# demos/overview/driver.ps1
# Automates demo requests for VHS screencast recording (Dashboard, Lazy-load, and LRU swap)

# Import shared helper utilities
$helpersPath = Join-Path $PSScriptRoot "..\common\driver_helpers.ps1"
if (Test-Path $helpersPath) {
    . $helpersPath
}

# 1. Wait for dashboard / server to be fully responsive
$serverUrl = "http://127.0.0.1:8081/v1/chat/completions"
$healthUrl = "http://127.0.0.1:8081/health"

if (Get-Command Wait-RuneServer -ErrorAction SilentlyContinue) {
    $null = Wait-RuneServer -Url $healthUrl -TimeoutSeconds 15
} else {
    Start-Sleep -Seconds 2
}

$headers = @{
    "Authorization" = "Bearer rune-demo-secure-key"
}

# 2. First Request: Lazy-load Qwen3.8 27B
$bodyQwen = @{
    model = "qwen3.8-27b"
    messages = @(
        @{ role = "user"; content = "Summarize the architectural philosophy of unified AI gateways in two sentences." }
    )
    max_tokens = 64
} | ConvertTo-Json

try {
    $null = Invoke-RestMethod -Uri $serverUrl -Method Post -Headers $headers -ContentType "application/json" -Body $bodyQwen -TimeoutSec 90
} catch {
    # Handled gracefully
}

# 3. Short 2.5s pause to observe completion, tok/s rate, and latency sparkline
Start-Sleep -Milliseconds 2500

# 4. Second Request: Ternary Bonsai 27B (triggers LRU model swap)
$bodyBonsai = @{
    model = "ternary-bonsai-2-27b-pq2-0"
    messages = @(
        @{ role = "user"; content = "Explain how LRU eviction prevents CUDA out-of-memory errors." }
    )
    max_tokens = 64
} | ConvertTo-Json

try {
    $null = Invoke-RestMethod -Uri $serverUrl -Method Post -Headers $headers -ContentType "application/json" -Body $bodyBonsai -TimeoutSec 90
} catch {
    # Handled gracefully
}

# 5. Hold final state
Start-Sleep -Seconds 6
