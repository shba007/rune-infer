# scripts/demo_driver.ps1
# Automates demo requests for VHS screencast recording

# 1. Brief pause for dashboard to appear on screen
Start-Sleep -Seconds 2

$serverUrl = "http://127.0.0.1:8081/v1/chat/completions"
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
    model = "ternary-bonsai-2-27b"
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
