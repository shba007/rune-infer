# Rune Infer

A standalone, config-driven, OpenAI-API-compatible model server written in Rust.

## Overview

Rune Infer serves multiple local and quantized models (GGUF, CACT, etc.) behind a unified HTTP server chosen per-request via the `model` ID. It features on-demand lazy loading, LRU memory eviction, pre-flight context/token budget validation, native multi-camera spatiotemporal video understanding, structured JSON schema decoding, and OpenAI-compatible tool/function calling.

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
                    ┌─────────────────────────────┐
                    │      Rune Infer (CLI)       │
                    │    clap-based entrypoint    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Config / Registry       │
                    │   models.json validation    │
                    │  VRAM estimation & budgets  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      HTTP Layer (Axum)      │
                    │  /v1/chat/completions       │
                    │  /v1/inference   /v1/models │
                    │  /health   Dual Log Writer  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Engine Registry / Router  │
                    │    Lazy Loader + LRU Cache  │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
│  Qwen2VlEngine  │       │  Qwen35Engine   │       │  NeedleEngine   │
│ (3D Tubelets &  │       │ (Hybrid Attention│      │(Tool Calling &  │
│    M-RoPE)      │       │   Text Model)   │       │ Structured JSON)│
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Project Layout

```text
rune-infer/
├── Cargo.toml            # Manifest & feature definitions (cpu, cuda, vulkan, metal)
├── config/
│   └── models.json       # Model catalog, runtime configs, and sampling parameters
├── logs/
│   └── rune-infer.log    # Persistent audit log (timestamps, status, latency, tokens)
├── src/
│   ├── main.rs           # Server bootstrap, CLI args, dual logger setup
│   ├── lib.rs            # Library entrypoint and public exports
│   ├── api.rs            # Axum router, OpenAI handlers, media processing
│   ├── config.rs         # ModelRegistry schema, capability resolution, token budgets
│   ├── types.rs          # OpenAI request/response structures, SSE, ChatMessage
│   └── inference/
│       ├── mod.rs        # AppState and engine dispatch
│       ├── registry.rs   # Engine catalog, VRAM calculation, LRU eviction
│       ├── traits.rs     # InferenceEngine trait
│       ├── types.rs      # Task request/response types
│       └── engines/
│           ├── bonsai.rs # Managed llama-server subprocess runner
│           ├── needle.rs # Cactus Needle 3 native engine
│           ├── qwen.rs   # LlamaModel text runner
│           ├── qwen2vl.rs# Multimodal vision/video engine (mtmd)
│           └── qwen35.rs # Qwen3.5 text engine
```

---

## Quick Start

### Prerequisites

- **Rust**: 1.80+ (Rust Edition 2024 compatible)
- **FFmpeg**: Must be available in `PATH` for video frame extraction and container decoding.
- **CUDA Toolkit** (Optional): For NVIDIA GPU acceleration.
- **C/C++ Compiler**: MSVC (Windows) or GCC/Clang (Linux/macOS) for native `llama.cpp` compilation.

### Building from Source

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
Usage: rune-infer [OPTIONS]

Options:
  -c, --config <CONFIG>            Path to models.json [default: config/models.json]
      --host <HOST>                Host IP to bind to [default: 0.0.0.0]
  -p, --port <PORT>                Port to bind to [default: from config]
      --max-loaded-models <N>      Max models kept in VRAM/RAM [default: 1]
      --idle-timeout <SECONDS>     Idle unload duration in seconds [default: 300]
  -h, --help                       Print help
  -V, --version                    Print version
```

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
2026-09-26T08:36:36.481767Z  INFO audit: Chat completion successful (text) status=200 latency_ms=80834 model=qwen3.8-27b prompt_tokens=2342 completion_tokens=973 media=images: 1, videos: 0
2026-09-26T08:48:30.698627Z  INFO audit: Chat completion successful (text) status=200 latency_ms=131529 model=qwen3.8-27b prompt_tokens=2675 completion_tokens=908 media=images: 0, videos: 1
2026-09-26T09:18:13.114716Z  WARN audit: Media decoding failed status=400 error=Remote server returned HTTP error for URL: HTTP status client error (403 Forbidden)
```

---

## Roadmap

### Phase 1: Text MVP (Current)
- [x] Config schema + loader + validation
- [x] `/health`, `/v1/models` endpoints
- [x] `TextEngine` wrapping llama-cpp-2
- [x] `/v1/chat/completions` streaming
- [x] Load on demand

### Phase 2: Vision
- [x] `VisionEngine` using mtmd
- [x] Image preprocessing
- [x] Vision-language generation

### Phase 3: Multi-model & LoRA
- [ ] LRU eviction
- [ ] Multiple concurrent models
- [ ] Per-request LoRA application

### Phase 4: Future Modalities
- [ ] TTS (Text-to-Speech)
- [ ] STT (Speech-to-Text)
- [ ] Image Generation

## Non-Goals

- Not a chat UI
- Not a training/fine-tuning tool
- Not a model downloader
- Not multi-tenant/auth-hardened for public internet

## License

MIT
