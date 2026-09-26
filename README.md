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

---

## API Reference & cURL Examples

All endpoints default to port `3423` (or as configured in `models.json`).

### 1. Health & Server Status

#### Check Server Health
```bash
curl -X GET http://localhost:3423/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "loaded_models": ["qwen3.8-27b"]
}
```

#### List Available Models
```bash
curl -X GET http://localhost:3423/v1/models
```

---

### 2. Standard Text Completions

#### Non-Streaming Chat Completion
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oxcoder-9b",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert Rust systems programmer."
      },
      {
        "role": "user",
        "content": "Explain zero-copy deserialization in serde."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 512
  }'
```

#### Server-Sent Events (SSE) Streaming
```bash
curl -N -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "user",
        "content": "Write a short poem about concurrent programming."
      }
    ],
    "stream": true
  }'
```

---

### 3. Structured JSON Mode (`response_format`)

Rune Infer natively supports OpenAI's `response_format` for models configured with `"Structured Output"` capability (e.g., `cactus-needle-3`, `qwen3.8-27b`).

#### Strict JSON Schema Mode
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "user_profile_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "full_name": { "type": "string" },
            "role": { "type": "string" },
            "skills": {
              "type": "array",
              "items": { "type": "string" }
            },
            "years_experience": { "type": "integer" }
          },
          "required": ["full_name", "role", "skills", "years_experience"]
        }
      }
    },
    "messages": [
      {
        "role": "user",
        "content": "Alex Vance is a Principal Systems Architect with 12 years working in Rust, Distributed Storage, and CUDA."
      }
    ],
    "temperature": 0.1
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-1790414800",
  "object": "chat.completion",
  "created": 1790414800,
  "model": "qwen3.8-27b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\n  \"full_name\": \"Alex Vance\",\n  \"role\": \"Principal Systems Architect\",\n  \"skills\": [\"Rust\", \"Distributed Storage\", \"CUDA\"],\n  \"years_experience\": 12\n}",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 142,
    "completion_tokens": 48,
    "total_tokens": 190
  }
}
```

#### Generic JSON Object Mode
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cactus-needle-3",
    "response_format": {
      "type": "json_object"
    },
    "messages": [
      {
        "role": "user",
        "content": "Extract the server IP and port from: Connection established to 192.168.1.100 on port 8080"
      }
    ]
  }'
```

---

### 4. Tool Calling (Function Calling)

#### Function Definition and Invocation
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "execute_shell_command",
          "description": "Execute a shell command on the host terminal",
          "parameters": {
            "type": "object",
            "properties": {
              "command": { "type": "string", "description": "The command line string to run" },
              "timeout_sec": { "type": "integer", "description": "Timeout in seconds" }
            },
            "required": ["command"]
          }
        }
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "Check the available disk space on the primary partition."
      }
    ],
    "temperature": 0.1
  }'
```

**Response with `tool_calls`:**
```json
{
  "id": "chatcmpl-1790415200",
  "object": "chat.completion",
  "created": 1790415200,
  "model": "qwen3.8-27b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_1790415200_0",
            "type": "function",
            "function": {
              "name": "execute_shell_command",
              "arguments": "{\"command\": \"df -h /\", \"timeout_sec\": 10}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 210,
    "completion_tokens": 32,
    "total_tokens": 242
  }
}
```

---

### 5. Multimodal Vision (Images)

Rune Infer accepts remote URLs, S3/storage links, base64 data URLs, and local file paths. Large images are automatically scaled according to the model's `max_resolution`.

#### Image via Remote / Object Storage URL
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-1.5-9b",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Analyze the architecture diagram in this image." },
          {
            "type": "image_url",
            "image_url": { "url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png" }
          }
        ]
      }
    ]
  }'
```

#### Image via Base64 Data URL
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Identify what is in this image." },
          {
            "type": "image_url",
            "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBD..." }
          }
        ]
      }
    ]
  }'
```

---

### 6. Native Video Understanding (`video_url`)

Rune Infer handles video natively. Remote video URLs (`.mp4`, `.webm`, `.mov`, `.mkv`) are streamed directly to disk, decoded via FFmpeg, converted into 3D spatiotemporal tubelet tokens, and wrapped in Qwen's native video conversation format (`Video 1: <|video_start|>...<|video_end|>`).

#### Video via Remote HTTPS / Presigned S3 URL
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Summarize the key events in this video in 3 bullet points with timestamps."
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
            }
          }
        ]
      }
    ],
    "temperature": 0.4,
    "max_tokens": 512
  }'
```

#### Video Analysis with Structured JSON Extraction
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "video_temporal_events",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "video_summary": { "type": "string" },
            "timeline": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "timestamp": { "type": "string" },
                  "action": { "type": "string" }
                },
                "required": ["timestamp", "action"]
              }
            }
          },
          "required": ["video_summary", "timeline"]
        }
      }
    },
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Extract all temporal action markers from this recording." },
          {
            "type": "video_url",
            "video_url": { "url": "http://127.0.0.1:9000/videos/factory_clip.mp4" }
          }
        ]
      }
    ]
  }'
```

---

### 7. RustFS / S3 Object Storage Workflow

For large media files that cannot be sent via Base64, run the included `docker-compose.yml` to spin up a local S3-compatible RustFS bucket service:

```bash
docker compose up -d
```

#### 1. Create a Bucket (using `curl --aws-sigv4`)
```bash
curl -X PUT \
  --aws-sigv4 "aws:amz:us-east-1:s3" \
  --user "rustfsadmin:rustfsadmin" \
  http://localhost:9000/videos
```

#### 2. Stream Upload Video Binary to S3/RustFS
```bash
curl -X PUT \
  --aws-sigv4 "aws:amz:us-east-1:s3" \
  --user "rustfsadmin:rustfsadmin" \
  -H "Content-Type: video/mp4" \
  -T "local_recording.mp4" \
  http://localhost:9000/videos/sample.mp4
```

#### 3. Run Inference against the Stored Asset
```bash
curl -X POST http://localhost:3423/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Describe the main activity in this video." },
          {
            "type": "video_url",
            "video_url": { "url": "http://127.0.0.1:9000/videos/sample.mp4" }
          }
        ]
      }
    ]
  }'
```

---

### 8. Direct Engine Task Execution (`/v1/inference`)

For direct task invocation or benchmarking bypass of the OpenAI conversation layer:

```bash
curl -X POST "http://localhost:3423/v1/inference?engine=cactus-needle-3" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "tool_call",
    "prompt": "Schedule a meeting with David tomorrow at 2pm",
    "schema": {
      "type": "object",
      "properties": {
        "attendee": { "type": "string" },
        "time": { "type": "string" }
      },
      "required": ["attendee", "time"]
    }
  }'
```

---

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
