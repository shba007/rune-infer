# demos/common/driver_helpers.ps1
# Shared helper functions for VHS demo simulation drivers

<#
.SYNOPSIS
    Waits for the Rune Infer server to become responsive at its /health endpoint.
#>
function Wait-RuneServer {
    param(
        [string]$Url = "http://127.0.0.1:8081/health",
        [int]$TimeoutSeconds = 20,
        [int]$PollIntervalMs = 250
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($response -and ($response.status -eq "ok" -or $response.status -eq "healthy" -or $response.version)) {
                return $true
            }
            return $true
        } catch {
            Start-Sleep -Milliseconds $PollIntervalMs
        }
    }
    return $false
}

<#
.SYNOPSIS
    Sends an authenticated JSON POST request to the Rune Infer server.
#>
function Invoke-DemoPost {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [object]$Payload,
        [string]$ApiKey = "rune-demo-secure-key",
        [int]$TimeoutSec = 90
    )

    $headers = @{
        "Authorization" = "Bearer $ApiKey"
    }

    $json = if ($Payload -is [string]) { $Payload } else { $Payload | ConvertTo-Json -Depth 10 }

    try {
        return Invoke-RestMethod -Uri $Url -Method Post -Headers $headers -ContentType "application/json" -Body $json -TimeoutSec $TimeoutSec -ErrorAction Stop
    } catch {
        # Catch and suppress if server is exiting during shutdown
        return $null
    }
}
