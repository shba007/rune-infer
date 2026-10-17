# Rune Infer VHS Screencasts & Demos

This directory contains automated terminal screencast recordings and background workload drivers for [Rune Infer](https://github.com/shba007/rune-infer). The recordings are captured using [Charm VHS](https://github.com/charmbracelet/vhs).

---

## Directory Structure

```text
demos/
├── README.md                  # This guide and demo catalog
├── record.ps1                 # Master orchestrator script
├── common/
│   ├── driver_helpers.ps1     # Shared background driver utilities (Wait-RuneServer, etc.)
│   └── template.tape          # Starter VHS tape template for new recordings
├── overview/
│   ├── overview.tape          # 60s screencast: live dashboard, lazy-loading & VRAM LRU swap
│   └── driver.ps1             # Background driver issuing chat completion requests
└── tts/
    ├── tts.tape               # Text-to-speech screencast: Audio8 real-time synthesis
    └── driver.ps1             # Background driver issuing /v1/audio/speech request
```

---

## Catalog of Recordings

| Demo | Tape File | Output Asset | Description |
| :--- | :--- | :--- | :--- |
| **Overview** | [`demos/overview/overview.tape`](overview/overview.tape) | [`docs/demo.gif`](../docs/demo.gif) | Live dashboard with 0 models, lazy-loading Qwen3.8 27B, LRU model swap to Ternary Bonsai 27B within VRAM budget, tok/s throughput, and latency sparkline. Embedded in main README. |
| **TTS** | [`demos/tts/tts.tape`](tts/tts.tape) | [`docs/demo_tts.gif`](../docs/demo_tts.gif) | Real-time text-to-speech synthesis using Audio8 TTS Preview 0.6B on `/v1/audio/speech`. |

---

## Prerequisites

1. **Charm VHS**:
   Install via WinGet, Scoop, or Go:
   ```powershell
   winget install charmbracelet.vhs
   # or
   scoop install vhs
   ```
2. **Rust & Release Binary**:
   The demos run against `target/release/rune-infer.exe` (or `rune-infer` on Linux/macOS) with `config/models.json`.

---

## Recording Demos

Use the included orchestrator script `demos/record.ps1`:

```powershell
# Validate all tapes without rendering
.\demos\record.ps1 -ValidateOnly

# Record a specific demo
.\demos\record.ps1 -Name overview
.\demos\record.ps1 -Name tts

# Compile release binary and record all demos
.\demos\record.ps1 -Build

# Or run VHS directly from the repository root
vhs demos/overview/overview.tape
vhs demos/tts/tts.tape
```

---

## Creating a New Demo

To add a new recording for another feature or modality:

1. **Create the folder**:
   ```powershell
   New-Item -ItemType Directory -Path "demos/<feature>"
   ```
2. **Create the tape script**:
   Copy [`demos/common/template.tape`](common/template.tape) into `demos/<feature>/<feature>.tape` and customize:
   - Output path: `Output docs/demo_<feature>.gif`
   - Background driver path: `.\demos\<feature>\driver.ps1`
   - Sleep durations matching the expected request runtime.
3. **Create the background driver**:
   Create `demos/<feature>/driver.ps1`:
   ```powershell
   # Import shared helper utilities
   $helpersPath = Join-Path $PSScriptRoot "..\common\driver_helpers.ps1"
   if (Test-Path $helpersPath) { . $helpersPath }

   # Wait for server /health to be responsive
   $null = Wait-RuneServer -Url "http://127.0.0.1:8081/health" -TimeoutSeconds 15

   # Issue your demo request(s)
   Invoke-RestMethod -Uri "http://127.0.0.1:8081/v1/..." ...
   ```
4. **Validate and record**:
   ```powershell
   .\demos\record.ps1 -Name <feature> -ValidateOnly
   .\demos\record.ps1 -Name <feature>
   ```
5. **Update documentation**:
   Add the new demo to the table above and document the output in [`docs/README.md`](../docs/README.md).
