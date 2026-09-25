# Rune Infer

A standalone, config-driven, OpenAI-API-compatible model server written in Rust.

## Overview

Rune Infer serves multiple local models (different weights, different quantizations) behind one HTTP server, chosen per-request via `model` id — similar in spirit to `llama-server`/Ollama/LM Studio, but config-file-first and designed from day one to grow into a general local-inference hub.

## Architecture

### High-Level Design

```
                    ┌─────────────────────────────┐
                    │        Rune Infer (CLI)         │
                    │  clap-based entrypoint        │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      Config Loader/Watcher      │
                    │  models.json -> ModelRegistry   │
                    │  (serde, validated, hot-reload) │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │        HTTP Layer (axum)        │
                    │  /v1/chat/completions           │
                    │  /v1/models  /health             │
                    │  OpenAI request/response types   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     Engine Manager / Router      │
                    │  - resolves model id -> config   │
                    │  - owns loaded-engine cache       │
                    │  - load/unload/evict policy       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     Engine (llama-cpp-2)        │
                    │  TextEngine + VisionEngine     │
                    └─────────────────────────────────┘
```

### Crate Structure

```
Rune Infer/
├── Cargo.toml                # workspace root
├── crates/
│   ├── Rune Infer-cli/          # Binary: CLI args, main(), wiring
│   ├── Rune Infer-config/       # Config schema, serde types, validation, hot-reload
│   ├── Rune Infer-api/          # OpenAI-compatible request/response types + axum handlers
│   ├── Rune Infer-core/         # Engine trait, ModelRegistry, EngineManager/router, errors
│   └── Rune Infer-engine-llama/ # llama-cpp-2 wrapper: TextEngine + VisionEngine
├── config/
│   └── models.json           # Example / default config
├── AGENTS.md
└── README.md
```

## Quick Start

### Prerequisites

- Rust 1.75+
- CUDA toolkit (for GPU acceleration on Windows/Linux)
- Visual Studio Build Tools or GCC/Clang (for C/C++ toolchain)

### Building

#### CPU-only (default)

```bash
cargo run --release --features cpu
```

#### CUDA (Windows/Linux)

```bash
cargo run --release --features cuda
```

#### Vulkan (Linux)

```bash
cargo run --release --features vulkan
```

#### Metal (macOS)

```bash
cargo run --release --features metal
```

### Running

```bash
cargo run --release -- --config config/models.json
```

Or with specific GPU backend:

```bash
cargo run --release --features cuda -- --config config/models.json
```
## Downloading Inference engines
Here are unified commands for each platform. 

For **Windows (CUDA)**, the command downloads and extracts **both** the executables (`llama-server.exe`, etc.) and the CUDA runtime libraries (`cudart*.dll`, `cublas*.dll`) into the same target folder in a single step.

---

### 1. Windows (CUDA / NVIDIA GPU)

> Downloads **both** the binary archive and the `cudart` runtime archive, extracting everything into `bin/llama-prism-latest-win-cuda/`.

#### Bash / Git Bash:
```bash
mkdir -p bin/llama-prism-latest-win-cuda && \
curl -s https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest \
  | grep -o 'https://[^"]*bin-win-cuda[^"]*\.zip' \
  | while read -r url; do \
      echo "--> Fetching: $url"; \
      curl -L "$url" -o bin/temp.zip && \
      tar -xf bin/temp.zip -C bin/llama-prism-latest-win-cuda && \
      rm bin/temp.zip; \
    done && \
echo "Done! Verifying:" && ls -l bin/llama-prism-latest-win-cuda/*.exe
```

#### Native PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "bin\llama-prism-latest-win-cuda" | Out-Null
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest"
$release.assets | Where-Object { $_.name -match "bin-win-cuda" } | ForEach-Object {
    Write-Host "--> Downloading: $($_.name)"
    $zipPath = "bin\temp.zip"
    Invoke-WebRequest -Uri $_.browser_download_url -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath "bin\llama-prism-latest-win-cuda" -Force
    Remove-Item $zipPath
}
Write-Host "Done! Verifying:"
Get-ChildItem "bin\llama-prism-latest-win-cuda\*.exe"
```

---

### 2. Windows (Vulkan / AMD, Intel, or Universal GPU)

#### Bash / Git Bash:
```bash
mkdir -p bin/llama-prism-latest-win-vulkan && \
curl -s https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest \
  | grep -o 'https://[^"]*bin-win-vulkan[^"]*\.zip' \
  | head -n 1 \
  | while read -r url; do \
      echo "--> Fetching: $url"; \
      curl -L "$url" -o bin/temp.zip && \
      tar -xf bin/temp.zip -C bin/llama-prism-latest-win-vulkan && \
      rm bin/temp.zip; \
    done && \
echo "Done! Verifying:" && ls -l bin/llama-prism-latest-win-vulkan/*.exe
```

#### Native PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "bin\llama-prism-latest-win-vulkan" | Out-Null
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest"
$asset = $release.assets | Where-Object { $_.name -match "bin-win-vulkan" } | Select-Object -First 1
Write-Host "--> Downloading: $($asset.name)"
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile "bin\temp.zip"
Expand-Archive -Path "bin\temp.zip" -DestinationPath "bin\llama-prism-latest-win-vulkan" -Force
Remove-Item "bin\temp.zip"
Get-ChildItem "bin\llama-prism-latest-win-vulkan\*.exe"
```

---

### 3. Linux (CUDA / NVIDIA GPU)

```bash
mkdir -p bin/llama-prism-latest-linux-cuda && \
curl -s https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest \
  | grep -o 'https://[^"]*bin-[^"]*cuda[^"]*\.tar\.gz' \
  | head -n 1 \
  | while read -r url; do \
      echo "--> Fetching: $url"; \
      curl -L "$url" -o bin/temp.tar.gz && \
      tar -xzf bin/temp.tar.gz -C bin/llama-prism-latest-linux-cuda --strip-components=1 2>/dev/null || tar -xzf bin/temp.tar.gz -C bin/llama-prism-latest-linux-cuda && \
      rm bin/temp.tar.gz; \
    done && \
echo "Done! Verifying:" && ls -l bin/llama-prism-latest-linux-cuda/llama-server
```

---

### 4. macOS (Apple Silicon / Metal)

```bash
mkdir -p bin/llama-prism-latest-macos && \
curl -s https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest \
  | grep -o 'https://[^"]*bin-macos-arm64[^"]*\.tar\.gz' \
  | head -n 1 \
  | while read -r url; do \
      echo "--> Fetching: $url"; \
      curl -L "$url" -o bin/temp.tar.gz && \
      tar -xzf bin/temp.tar.gz -C bin/llama-prism-latest-macos --strip-components=1 2>/dev/null || tar -xzf bin/temp.tar.gz -C bin/llama-prism-latest-macos && \
      rm bin/temp.tar.gz; \
    done && \
echo "Done! Verifying:" && ls -l bin/llama-prism-latest-macos/llama-server
```

---

### Verification

After running the Windows CUDA command, your `bin/llama-prism-latest-win-cuda` directory will contain both the binaries and DLLs:

```text
bin/llama-prism-latest-win-cuda/
├── llama-server.exe
├── llama-cli.exe
├── ggml.dll
├── llama.dll
├── cublas64_12.dll
├── cublasLt64_12.dll
└── cudart64_12.dll
```

Run:
```bash
./bin/llama-prism-latest-win-cuda/llama-server.exe --version
```

## Downloading Inference Models

The inference engines (needle, qwen3, qwen35) auto-discover their weights from
the `weights/` folder at startup. Place a GGUF file there and the matching
engine loads it on first start — no config change needed.

### Qwen3.5-0.8B (16-bit / BF16)

```bash
curl -L -o weights/Qwen3.5-0.8B-BF16.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-BF16.gguf"
```

The `-L` flag follows the redirect to Hugging Face's CDN. The engine
(`qwen35-0.8b`) picks it up automatically.

## API

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "loaded_models": ["oxcoder-9b"]
}
```

#### `GET /v1/models`

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "oxcoder-9b",
      "object": "model",
      "owned_by": "Rune Infer"
    }
  ]
}
```

#### `POST /v1/chat/completions`

Chat completion endpoint (OpenAI-compatible).

**Request:**
```json
{
  "model": "oxcoder-9b",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Streaming Response:**
```
data: {"choices":[{"index":0,"delta":{"content":"Paris"}}]}

data: {"choices":[{"index":0,"delta":{"content":" is"}}]}

data: {"choices":[{"index":0,"delta":{"content":" the"}}]}

data: {"choices":[{"index":0,"delta":{"content":" capital"}}]}

data: {"choices":[{"index":0,"delta":{"content":" of"}}]}

data: {"choices":[{"index":0,"delta":{"content":" France"}}]}

data: {"choices":[{"index":0,"delta":{}}}

data: [DONE]
```

### Vision Models

Vision models support image inputs via `image_url`:

```json
{
  "model": "ornith-1.5-9b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBOR..."
          }
        }
      ]
    }
  ]
}
```

## Configuration

Config file (`config/models.json`):

```json
{
  "schema_version": 1,
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "api_key": null,
    "max_loaded_models": 1,
    "idle_unload_seconds": 300
  },
  "models": [
    {
      "id": "oxcoder-9b",
      "name": "OxCoder 9B",
      "architecture": "qwen35",
      "format": "gguf",
      "model_path": "D:/Models/...",
      "mmproj_path": null,
      "vision": false,
      "modality": "text",
      "description": "Code generation",
      "sampling": {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.01,
        "max_tokens": 2048
      },
      "runtime": {
        "context_length": 32768,
        "gpu_layers": 99,
        "n_threads": 4,
        "extra_args": ""
      },
      "lora": []
    }
  ]
}
```

### CLI Options

```
Usage: rune-infer [OPTIONS]

Options:
  -c, --config <CONFIG>    Path to configuration file [default: config/models.json]
  -h, --host <HOST>        Host to bind to [default: 0.0.0.0]
  -p, --port <PORT>        Port to bind to [default: 8080]
  -C, --cuda               Enable CUDA backend
  -V, --vulkan             Enable Vulkan backend
  -M, --metal              Enable Metal backend
  -c, --cpu                CPU-only mode
  -w, --watch-config       Enable config hot-reload [default: true]
  -k, --api-key <KEY>      API key for authentication
      --max-loaded-models <N>    Maximum number of loaded models [default: 1]
      --idle-timeout <SECONDS>   Idle unload timeout [default: 300]
  -h, --help               Print help
  -V, --version            Print version
```

## Roadmap

### Phase 1: Text MVP (Current)
- [x] Config schema + loader + validation
- [x] `/health`, `/v1/models` endpoints
- [ ] `TextEngine` wrapping llama-cpp-2
- [ ] `/v1/chat/completions` streaming
- [ ] Load on demand

### Phase 2: Vision
- [ ] `VisionEngine` using mtmd
- [ ] Image preprocessing
- [ ] Vision-language generation

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
