# Rune Infer — Feature Todo

## 1. Configuration & Migration
- [x] Schema 3 decoupled configuration files (`server.json` and `catalog.json`)
- [x] Environment variable templating (`{{VAR_NAME}}`) resolved via `.env` and system environment
- [x] Strict unknown-field validation denying unrecognized keys in catalog and server configs
- [x] CLI migration tool (`rune-infer migrate`) upgrading Schema 1 and Schema 2 to Schema 3 with automated backup creation (`.v1.bak` / `.v2.bak`)
- [x] Runtime filesystem polling hot-reload (`spawn_config_watcher`) detecting configuration changes every 2 seconds without restarting
- [x] Home workspace directory resolution (`--config` > `RUNE_HOME` > dev `./config` > `~/.rune/infer`)
- [ ] Configuration hot-reload via filesystem events (`notify` crate) instead of 2-second polling
- [ ] Model catalog dynamic update via REST API without file editing

## 2. Host Safety, Budgeting & Lifecycle
- [x] Pre-flight token calculation preventing GGML/CUDA out-of-memory aborts (`SIGABRT`)
- [x] Multimodal token budgeting for visual patch tokens (28×28) and 3D video tubelets
- [x] GPU VRAM budget ceiling enforcement (`vram_budget_ratio`, default 95%)
- [x] Single and multi-model LRU eviction when exceeding loaded model limits (`max_loaded_models`)
- [x] Background idle model eviction timer (`evict_idle`, `--idle-timeout`, default 300s)
- [x] OS-level subprocess lifecycle supervision (Windows Job Objects / Unix `PR_SET_PDEATHSIG` and `SIGKILL`)
- [x] ModelExecutionGate with concurrency gating and priority queuing
- [x] Studio priority contract: foreground requests yield running background jobs with HTTP 503 or SSE error events
- [ ] Per-model concurrent execution slots (parallel multi-request slots per model)
- [ ] Per-request LoRA weight dynamic swapping (parsed and validated in config, but not yet applied by engines)
- [ ] Decode-only token generation throughput benchmark (currently measures prompt processing + decode)

## 3. Subprocess Engine Orchestration
- [x] Upstream `llama-server` managed subprocess engine for GGUF LLM and vision-language models
- [x] Ternary `llama-server-prism` engine for Prism 3-bit / ternary Bonsai models
- [x] `sd-server` (stable-diffusion.cpp) managed subprocess engine for text-to-image generation
- [x] `audio-cpp` managed subprocess engine for Audio8 speech synthesis models
- [x] `nemo-speech` managed subprocess engine for Parakeet / FastConformer speech models
- [x] `crispasr` managed subprocess engine for Whisper-family speech recognition
- [x] `whisper-cpp` managed subprocess engine for GGML Whisper models
- [x] `onnx` runtime engine for object detection and ADE20k scene segmentation
- [x] `litert` CompiledModel CPU and GPU adapters for `.tflite` multimodal vision models
- [x] Automatic runtime engine binary downloader and extractor with platform detection (`llama_server/install.rs`, `engines/download.rs`)
- [x] GGUF header sniffing and automatic model architecture/companion detection (`detect.rs`)
- [ ] Model weight automated pulling CLI (`rune-infer pull <model-id>`)
- [ ] Detailed engine startup and model loading progress reporting to TUI (currently binary loading/loaded state)
- [ ] Engine downloader support for offline native ARM64 fallback builds

## 4. Native Rust Engines
- [x] Native BERT-style safetensors `encoder` engine for embeddings, classification, and visual grounding
- [x] Native Cactus `needle` engine for `.cact` tool-calling models

## 5. Standard OpenAI & Anthropic API Surface
- [x] `POST /v1/chat/completions` (streaming SSE, non-streaming, multi-modal images, video tubelets, tool calling, JSON schema formatting)
- [x] `POST /v1/embeddings` (batch text embeddings and normalization)
- [x] `POST /v1/audio/transcriptions` (WAV audio speech-to-text with segment timestamps and speaker activity)
- [x] `POST /v1/audio/translations` (WAV audio translation to English)
- [x] `POST /v1/audio/speech` (TTS audio synthesis with customizable voice and speed)
- [x] `POST /v1/images/generations` (text-to-image generation with aspect ratio presets)
- [x] `POST /v1/images/edits` (instruction-guided image editing with optional PNG alpha masks)
- [x] `POST /v1/images/variations` (image variation generation with subtle composition shifts)
- [x] `POST /v1/moderations` (content moderation category scores)
- [x] `GET /v1/models` and `GET /v1/models/{model}` (catalog inspection with capability annotations)
- [x] `GET /health`, `GET /livez`, `GET /readyz` (liveness and readiness probes)
- [ ] Active routing for Anthropic Messages API (`/v1/messages` and `/anthropic/v1/messages`) (implemented in `src/api/anthropic.rs`, but commented out in `api.rs`)
- [ ] Active routing for OpenAI Responses API (`/v1/responses`) (implemented in `src/api/responses.rs`, but commented out in `api.rs`)

## 6. Rune-Specific Extended APIs
- [x] `POST /v1/inference` (direct task execution: depth estimation, embeddings, classification)
- [x] `POST /v1/nlu/intents` (intent classification and slot filling)
- [x] `POST /v1/ocr` (document text extraction and bounding box detection)
- [x] `POST /v1/decisions` (fast decision and probability score estimation)
- [x] `POST /v1/images/detections` (object detection with normalized bounding boxes)
- [x] `POST /v1/images/segmentations` (semantic scene segmentation and class map generation)
- [x] `POST /v1/images/portrait-mattings` (fine portrait alpha matte extraction)
- [x] `POST /v1/images/upscales` (2× and 4× super-resolution upscaling)
- [x] `POST /v1/images/captions` (visual image captioning)
- [x] `POST /v1/images/grounding` (visual object grounding from text queries)
- [x] `POST /v1/images/recognize` (visual object recognition)
- [x] `POST /v1/images/restorations` (image defect restoration)
- [x] `POST /v1/images/style-transfers` (visual style transfer)
- [x] `POST /v1/audio/diarizations` (speaker diarization intervals and turns)
- [x] `POST /v1/audio/speech-to-speech` (composite speech-to-speech conversion)
- [x] Management APIs (`/v1/management/roles`, `/v1/management/models/{id}/load`, `/v1/management/model-icon/{model}`, `/v1/capabilities`)
- [ ] Music generation engine backend (`music`)
- [ ] Vocal/instrument audio separation engine backend (`separate`)
- [ ] Zero-shot voice cloning and timbre conversion backends (`clone`, `convert`)
- [ ] Video generation engine backend (`video-create`)

## 7. Cloud Proxy & Provider Adapters
- [x] OpenAI adapter with rate limiting and retry handling
- [x] Anthropic adapter with Claude tool calling and vision support
- [x] Google Gemini adapter with multimodal vision and system instruction translation
- [x] OpenRouter adapter for cloud routing
- [x] Custom OpenAI-compatible upstream endpoint proxying
- [x] Token-bucket rate limiting (RPM, TPM, RPD) with concurrency caps and queue timeouts
- [ ] Multi-tenant per-client API keys with independent rate limit quotas

## 8. Terminal UX & Distribution
- [x] Interactive Ratatui TUI dashboard with live VRAM bar, tokens/second, latency charts, and audit logs
- [x] Interactive setup wizard (`rune-infer setup`) for network binding, API keys, and model catalog scanning
- [x] Headless mode (`--headless` or non-TTY detection)
- [x] Full-frame `NO_COLOR` and `RUNE_ASCII=1` fallback rendering
- [x] Executable self-update command (`rune-infer self-update`) with checksum verification and rollback
- [ ] Windows and macOS binary code signing (SmartScreen/Gatekeeper warnings)
- [ ] Docker container image packaging and publishing
- [ ] Uninstall script for shared `~/.rune` artifacts
