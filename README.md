# Rune Infer

A standalone, config-driven, OpenAI-API-compatible model server written in Rust.

## Overview

Rune Infer is a unified AI gateway: local models (GGUF, safetensors, CACT) and cloud providers (Gemini, OpenAI-compatible, Anthropic, OpenRouter) behind one HTTP endpoint, chosen per request via the `model` ID, across text, vision, audio, image, OCR and embedding modalities. It features on-demand lazy loading, LRU memory eviction, pre-flight context/token budget validation, native multi-camera spatiotemporal video understanding, structured JSON schema decoding, and OpenAI-compatible tool/function calling.

---

## Key Features

- **OpenAI-API Compatible**: Drop-in replacement for `/v1/chat/completions` (streaming & non-streaming), `/v1/models`, and `/health`.
- **Config-Driven & Lazy-Loaded**: Models are defined in `config/models.json` and loaded into VRAM/RAM only when requested, respecting `max_loaded_models` and LRU eviction.
- **Multimodal Vision & Native Video**:
  - Image inputs via HTTPS URLs, Base64 Data URLs, or local file paths.
  - Native video understanding using **3D Spatiotemporal Tubelets** and **M-RoPE** (continuous $(T, H, W)$ space-time coordinates).
  - Remote streaming from AWS S3, Cloudflare R2, MinIO, and RustFS (zero-RAM disk streaming).
- **Pre-Flight Context & Token Budget Protection**: Automatically calculates text and visual patch tokens against the context window before native execution, preventing native GGML memory aborts (`SIGABRT`).
- **Structured JSON Mode**: Full support for OpenAI `response_format` (`json_object` and `json_schema` with strict validation) for models with `Structured Output` capabilities (e.g., `cactus-needle-3`, `qwen3.8-27b`).
- **Tool Calling (Function Calling)**: Supports multi-tool definitions, schema validation, and streaming tool chunk generation.
- **Direct Task Inference**: `/v1/inference?engine=<id>` endpoint for raw task execution and structured extraction.
- **Auditing & Event Logging**: Lightweight structured request/response audit logging to `logs/rune-infer.log` with status, latency, media count, and token usage (payload bodies are omitted to prevent disk bloat).

---

## Architecture

```
 clap CLI ── .env ── config/models.json ({{ENV_VAR}} templating)
        │
   axum router ── auth (Bearer / x-api-key)
        │   /v1/chat/completions  /v1/responses  /v1/messages
        │   /v1/images/*  /v1/audio/*  /v1/embeddings  /v1/moderations
        │   /v1/nlu/intents  /v1/ocr  /v1/inference  /v1/models  /health
        │
   ┌────┴─────────────────────┐
   │ local model              │ remote model
   │ lazy load, VRAM budget,  │ proxy adapters (OpenAI, Anthropic, Gemini)
   │ LRU eviction             │ rate limit, queue, retries
   └────┬─────────────────────┘
        │ InferenceEngine trait
   llama-server (subprocess) · sd-server · crispasr · nemo_speech
   needle (cact) · encoder (safetensors)
```

See [AGENTS.md](AGENTS.md) for philosophy, scope and the full roadmap.

### Project Layout

```text
rune-infer/
├── config/models.json     # Model catalog, runtime configs, sampling
├── src/
│   ├── main.rs / lib.rs   # Bootstrap, CLI, logging
│   ├── api.rs, api/       # Router, auth, endpoint handlers (chat, audio, images, ...)
│   ├── config.rs          # Registry schema, validation, env templating, budgets
│   ├── media.rs           # Image/video input (URL, data URI, file, S3)
│   ├── types.rs           # OpenAI/Anthropic wire types
│   ├── inference/         # Registry (LRU, VRAM), process mgmt, engines/
│   └── proxy/             # Cloud provider adapters, rate limiting, retries
├── postman/               # API collection
└── tests/                 # Integration tests
```

---

## Install

**Linux / macOS**

```bash
curl -fsSL https://raw.githubusercontent.com/shba007/rune-infer/main/install.sh | sh
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/shba007/rune-infer/main/install.ps1 | iex
```

The installer downloads the release for your platform, verifies its SHA-256,
installs to `~/.local/bin` (or `%LOCALAPPDATA%\Programs\rune-infer`) and adds
it to your PATH. Re-run it to upgrade. Options (environment variables):
`RUNE_VERSION=v0.2.0`, `RUNE_INSTALL_DIR=...`, `RUNE_NO_MODIFY_PATH=1`.

Then just run:

```bash
rune-infer
```

On first run a **setup wizard** walks you through network/API-key settings,
optional cloud provider keys, optional local models and the inference engine
download. Everything is skippable and nothing is saved until the final review.
Your data lives in `~/.rune-infer` (`config/models.json`, `.env` for secrets,
`bin/` engines, `logs/`); override with `RUNE_HOME`. Re-run it any time with
`rune-infer setup`.

### Dashboard

The terminal dashboard shows each model's state (idle / loading / loaded / failed,
load time, VRAM), a live VRAM gauge, generation speed (tok/s) and latency
sparklines, recent requests and a filterable log view. Keys: `Tab` switch view,
`e` errors only, `?` help, `q` quit. For servers/containers use
`rune-infer --headless` (automatic when there is no terminal).

## Building from source

### Prerequisites

- **Rust**: toolchain supporting Edition 2024 (see `rust-version` in Cargo.toml)
- **FFmpeg**: Must be available in `PATH` for video frame extraction and container decoding.
- **CUDA Toolkit** (Optional): For NVIDIA GPU acceleration.
- **Engine binaries**: `llama-server` and friends are located via env (`LLAMA_SERVER_PATH`, `SD_SERVER_PATH`, `CRISPASR_PATH`, ...) or auto-installed.

### Build

```bash
# CPU Only
cargo build --release --features cpu

# CUDA (NVIDIA GPU Acceleration)
cargo build --release --features cuda

# Vulkan (AMD, Intel, or cross-platform GPU)
cargo build --release --features vulkan

# Metal (macOS Apple Silicon)
cargo build --release --features metal
```

### Running the Server

```bash
# Run with default config (config/models.json)
./target/release/rune-infer

# Specify custom config, host, port, and max loaded models
./target/release/rune-infer --config config/models.json --host 0.0.0.0 --port 3423 --max-loaded-models 2
```

### CLI Options

```text
Usage: rune-infer [OPTIONS] [COMMAND]

Commands:
  setup    Run the interactive setup wizard (API keys, local models, engines)

Options:
  -c, --config <CONFIG>        Path to models.json [default: ./config/models.json if present,
                               else ~/.rune-infer/config/models.json]
      --host <HOST>            Host IP to bind to [default: from config]
  -p, --port <PORT>            Port to bind to [default: from config]
      --max-loaded-models <N>  Models kept loaded at once, 0 = unlimited [default: from config]
      --idle-timeout <SECS>    Unload a model after this many idle seconds, 0 = never [default: from config]
      --headless               Plain server without the dashboard (implied without a TTY)
  -h, --help                   Print help
  -V, --version                Print version
```

Environment: `RUNE_HOME` relocates the data folder; secrets referenced in the
config as `"{{NAME}}"` are read from `.env` or the environment.

---

## Configuration (`models.json`)

The server configuration resides in `config/models.json`:

```json
{
  "schema_version": 1,
  "server": {
    "host": "0.0.0.0",
    "port": 3423,
    "api_key": null,
    "max_loaded_models": 1,
    "idle_unload_seconds": 300,
    "vram_budget_ratio": 0.97
  },
  "models": [
    {
      "id": "qwen3.8-27b",
      "name": "Qwen3.8 27B",
      "architecture": "qwen35",
      "format": "gguf",
      "model_path": "D:/Models/qwen3.8-27b.gguf",
      "mmproj_path": "D:/Models/mmproj-F16.gguf",
      "vision": true,
      "modality": "VisionText",
      "description": "Multimodal vision-language model",
      "max_resolution": "4096×4096 (Dynamic 4K)",
      "capabilities": [
        "Chat",
        "Vision",
        "Reasoning",
        "Agentic Tasks",
        "Tool Call",
        "Structured Output"
      ],
      "bits_per_weight": 3.44,
      "total_params": 27000000000,
      "max_context_length": 262144,
      "sampling": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "max_tokens": 4096
      },
      "runtime": {
        "context_length": -1,
        "gpu_layers": 99,
        "extra_args": ""
      }
    }
  ]
}
```

## Logging & Auditing

Rune Infer writes dual output:
- **Console (`stdout`)**: Real-time server diagnostics and tracing.
- **Persistent Audit Log (`logs/rune-infer.log`)**: Compact audit trail documenting every transaction without storing large payload bodies:

```text
2026-09-26T08:29:38.272845Z  INFO rune_infer: Starting Rune Infer...
2026-09-26T08:29:38.273508Z  INFO rune_infer: Server listening on 0.0.0.0:3423
2026-09-26T08:36:36.481767Z  INFO audit: Request completed event="completed" status=200 model=qwen3.8-27b latency_ms=80834 prompt_tokens=2342 completion_tokens=973 generation_ms=61200
2026-09-26T08:48:30.698627Z  INFO audit: Request completed event="completed" status=200 model=qwen3.8-27b latency_ms=131529 prompt_tokens=2675 completion_tokens=908 generation_ms=97400
2026-09-26T09:18:13.114716Z  WARN audit: Media decoding failed status=400 error=Remote server returned HTTP error for URL: HTTP status client error (403 Forbidden)
```

---

## Roadmap

See [AGENTS.md](AGENTS.md#7-roadmap-priority-order) for details.

1. Stability & test hardening
2. More modalities/models
3. Gateway features (usage/cost metrics, caching, rate limits)
4. Packaging & distribution (binaries, Docker, model `pull`)
5. Per-request LoRA, multi-model concurrency, idle unload
6. Config hot-reload and config splitting

## Non-Goals

- Not a chat UI, training tool, or agent/orchestration loop
- Not a distributed server
- Not hardened for multi-tenant public internet

Secrets belong in `.env`; reference them in config as `"{{GEMINI_API_KEY}}"`.

## License

MIT
