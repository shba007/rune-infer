# Rune Infer VHS Screencasts & Demos

This directory contains automated terminal screencast recordings and background workload drivers for [Rune Infer](https://github.com/shba007/rune-infer). The recordings are captured using [Charm VHS](https://github.com/charmbracelet/vhs).

---

## Directory Structure

```text
demos/
├── README.md                  # This guide and demo catalog
├── record.sh                  # Master orchestrator script
├── common/
│   ├── driver_helpers.sh      # Shared background driver utilities (wait_rune_server, etc.)
│   └── template.tape          # Starter VHS tape template for new recordings
├── overview/
│   ├── overview.tape          # 60s screencast: live dashboard, lazy-loading & VRAM LRU swap
│   └── driver.sh              # Background driver issuing chat completion requests
└── tts/
    ├── tts.tape               # Text-to-speech screencast: Audio8 real-time synthesis
    └── driver.sh              # Background driver issuing /v1/audio/speech request
```

---

## Catalog of Recordings

| Demo | Tape File | Output Asset | Description |
| :--- | :--- | :--- | :--- |
| **Overview** | [`demos/overview/overview.tape`](overview/overview.tape) | [`public/demo.gif`](../public/demo.gif) | Live dashboard with 0 models, lazy-loading Qwen3.8 27B, LRU model swap to Ternary Bonsai 27B within VRAM budget, tok/s throughput, and latency sparkline. Embedded in main README. |
| **TTS** | [`demos/tts/tts.tape`](tts/tts.tape) | [`public/demo_tts.gif`](../public/demo_tts.gif) | Real-time text-to-speech synthesis using Audio8 TTS Preview 0.6B on `/v1/audio/speech`. |

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

Use the included orchestrator script `demos/record.sh`:

```bash
# Validate all tapes without rendering
bash demos/record.sh --validate-only

# Record a specific demo
bash demos/record.sh overview
bash demos/record.sh tts

# Compile release binary and record all demos
bash demos/record.sh --build

# Or run VHS directly from the repository root
vhs demos/overview/overview.tape
vhs demos/tts/tts.tape
```

---

## Creating a New Demo

To add a new recording for another feature or modality:

1. **Create the folder**:
   ```bash
   mkdir -p demos/<feature>
   ```
2. **Create the tape script**:
   Copy [`demos/common/template.tape`](common/template.tape) into `demos/<feature>/<feature>.tape` and customize:
   - Output path: `Output docs/demo_<feature>.gif`
   - Background driver path: `demos/<feature>/driver.sh`
   - Sleep durations matching the expected request runtime.
3. **Create the background driver**:
   Create `demos/<feature>/driver.sh`:
   ```bash
   #!/usr/bin/env bash
   # Import shared helper utilities
   here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   . "$here/../common/driver_helpers.sh"

   # Wait for server /health to be responsive
   wait_rune_server "http://127.0.0.1:8081/health" 15

   # Issue your demo request(s)
   demo_post "http://127.0.0.1:8081/v1/..." '{ ... }' >/dev/null
   ```
4. **Validate and record**:
   ```bash
   bash demos/record.sh <feature> --validate-only
   bash demos/record.sh <feature>
   ```
5. **Update documentation**:
   Add the new demo to the table above and document the output in [`docs/README.md`](../docs/README.md).
