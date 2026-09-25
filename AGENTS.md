A standalone, config-driven, OpenAI-API-compatible model server written in
Rust. It serves multiple local models (different weights, different
quantizations) behind one HTTP server, chosen per-request via `model` id —
similar in spirit to `llama-server`/Ollama/LM Studio, but config-file-first
and designed from day one to grow into a general local-inference hub
(text, vision, TTS, STT, image generation, LoRA pipelines).

This document is the source of truth for scope, architecture, and phased
delivery. Update it whenever a decision below changes.

---

## 1. Decisions locked in for the MVP

| Question | Decision |
|---|---|
| Inference backend | Rust FFI bindings to `llama.cpp` via the `llama-cpp-2` crate (`utilityai/llama-cpp-rs`). Not a subprocess wrapper — we link `libllama`/`libmtmd` directly. |
| Vision | Use `llama-cpp-2`'s `mtmd` module (`MtmdInputChunk`, `MtmdInputChunks`) for image+text models that ship an `mmproj` file. |
| Model formats | GGUF only for MVP. Safetensors (via `candle` or similar) is a post-MVP milestone — see §7. |
| Platform | Cross-platform from day one: Windows + Linux, with GPU backend selected at **build time** via Cargo features (`cuda`, `vulkan`, `metal`, `cpu`), matching how `llama-cpp-2`/`llama.cpp` picks backends. One binary per backend combo; document the build matrix. |
| API surface | OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming), image_url content parts for vision. `/v1/models` for discovery. `/v1/completions` and `/v1/embeddings` are stretch goals, not blocking. |
| Model swapping | Config-driven, hot-reloadable model *registry* (what models exist and their settings) is P0. Actually swapping which model is *loaded into VRAM* at request time (multi-model concurrency / LRU eviction) is P1 — MVP can start with "one loaded model at a time, switch on demand" and be explicit about that limitation. |
| Orchestration | This is a **model server**, not an agent/harness. No tool-calling loop, no planning, no multi-step execution. It exposes inference; callers (e.g. Claude Code style clients, custom apps) drive the loop. |
| Future scope (not MVP, but the config schema & crate boundaries must not preclude them) | TTS, STT, image generation (e.g. z-image + custom LoRA, MiniMax H3-style models), and ComfyUI-like *pipelines* expressed purely as config (chaining model stages, swapping LoRA per request). |

---

## 2. Non-goals (explicitly out of scope)

- Not a chat UI. No frontend beyond maybe a minimal `/health` and `/v1/models` JSON.
- Not a training/fine-tuning tool.
- Not a model downloader/hub client in the MVP (models are already on disk; paths come from config). A `pull` helper can come later.
- Not multi-tenant/auth-hardened for the public internet in the MVP — assume trusted local/LAN use, but leave a hook for an API-key header check.
- Not a distributed/multi-node server. Single process, single machine.

---

## 3. High-level architecture

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
        ┌───────────────────────────┼───────────────────────────┐
        │                            │                            │
┌───────▼────────┐         ┌────────▼────────┐          ┌───────▼────────┐
│ TextEngine      │         │ VisionEngine     │          │ (future)       │
│ llama-cpp-2      │         │ llama-cpp-2+mtmd  │          │ TtsEngine,     │
│ LlamaContext     │         │ image preprocess  │          │ SttEngine,     │
│ sampling/stream  │         │ + text generation  │          │ ImageGenEngine │
└──────────────────┘         └──────────────────┘          └────────────────┘
```

### Key architectural principle: **Engine trait, not a giant match statement**

Every backend (text-llama, vision-llama, later TTS/STT/imagegen) implements a
common `Engine` trait so the HTTP layer and router never need to know backend
internals:

```rust
#[async_trait]
pub trait Engine: Send + Sync {
    fn id(&self) -> &str;
    fn modality(&self) -> Modality; // Text, VisionText, Tts, Stt, ImageGen, ...
    async fn generate(&self, req: GenerationRequest) -> Result<GenerationStream>;
    fn is_loaded(&self) -> bool;
    async fn unload(&self) -> Result<()>;
    fn memory_estimate_bytes(&self) -> Option<u64>;
}
```

This is the single most important design decision for future-proofing: the
router, HTTP handlers, and config loader depend only on this trait and on
the config schema — never on `llama-cpp-2` types directly outside the
`engines::llama` module. When TTS/STT/imagegen engines are added later, they
are new modules implementing the same trait; nothing upstream changes.

---

## 4. Crate/module layout

```
Rune Infer/
├── Cargo.toml                # workspace root
├── crates/
│   ├── Rune Infer-cli/          # binary crate: clap args, main(), wiring
│   ├── Rune Infer-config/       # config schema, serde types, validation, hot-reload
│   ├── Rune Infer-api/          # OpenAI-compatible request/response types + axum handlers
│   ├── Rune Infer-core/         # Engine trait, ModelRegistry, EngineManager/router, errors
│   ├── Rune Infer-engine-llama/ # llama-cpp-2 wrapper: TextEngine + VisionEngine (mtmd)
│   └── Rune Infer-engine-*/     # (future) tts, stt, imagegen — same Engine trait
├── config/
│   └── models.json           # example / default config
├── AGENTS.md
└── README.md
```

Rationale: splitting `Rune Infer-engine-llama` from `Rune Infer-core` means the
`unsafe` FFI surface (llama-cpp-2 is explicitly not memory-safe per its own
docs) is quarantined to one crate, and a future safetensors/candle engine or
a TTS engine doesn't need to touch it at all.

---

## 5. Config schema (v1, extensible)

Config is the product's spine — everything (model swapping, LoRA, future
pipelines) is expressed here rather than in code. Design it to version
cleanly.

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
      "modality": "text",
      "architecture": "qwen35",
      "format": "gguf",
      "model_path": "D:/Models/.../OxCoder-9B.i1-Q6_K.gguf",
      "mmproj_path": null,
      "vision": false,
      "description": "Code generation and instruction following",
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
        "extra_args": ""
      },
      "lora": []
    },
    {
      "id": "qwen2vl-7b",
      "name": "Qwen2-VL 7B",
      "modality": "vision-text",
      "architecture": "qwen2vl",
      "format": "gguf",
      "model_path": "D:/Models/qwen2vl-7b-q4_k_m.gguf",
      "mmproj_path": "D:/Models/qwen2vl-7b-mmproj-f16.gguf",
      "vision": true,
      "sampling": { "temperature": 0.7, "top_p": 0.9, "max_tokens": 1024 },
      "runtime": { "context_length": 8192, "gpu_layers": 99 }
    }
  ]
}
```

Design notes:
- `modality` replaces the implicit `vision: bool` as the extensible field
  (`text`, `vision-text`, `tts`, `stt`, `image-gen`, ...); keep `vision`
  as a derived/back-compat convenience if you want, but the engine
  dispatch switches on `modality`.
- `format`: `"gguf"` for MVP; reserved values `"safetensors"`, `"onnx"` for
  later so old configs don't need a breaking migration.
- `lora`: empty array for MVP but present in the schema now — an array of
  `{ path, scale }` objects. Wiring this to `llama-cpp-2`'s LoRA management
  API (mentioned in its docs) is a good P1 task since the crate already
  supports it.
- `schema_version`: bump and write a small migration function whenever the
  shape changes — do this from day one, it's cheap now and painful later.
- Config is loaded once at startup and validated (paths exist, ids unique,
  architecture/modality combinations sane); file-watching for hot-reload of
  the *registry* (add/remove/edit model entries) is a P1 nice-to-have via
  `notify` crate — does not require unloading a running engine unless that
  specific model's entry changed.

---

## 6. API behavior (MVP)

- `GET /v1/models` → lists configured models (id, name, modality) regardless
  of load state.
- `POST /v1/chat/completions` → OpenAI-shaped body. `model` field selects
  config entry. Supports:
  - text-only messages
  - `content: [{type:"text",...}, {type:"image_url",...}]` for vision models
  - `stream: true` via SSE, matching OpenAI's `data: {...}\n\n` framing and
    final `data: [DONE]`
  - per-request sampling overrides layered on top of the config's
    `sampling` defaults
- `GET /health` → process + loaded-model status.
- Error shape mirrors OpenAI's `{"error": {"message", "type", "code"}}` so
  existing OpenAI-client SDKs work unmodified against this server.
- On a chat request for a model that isn't currently loaded: load it
  (evicting per `max_loaded_models` / LRU if needed), then serve. First
  request to a "cold" model will be slow (model load time) — document this,
  consider an optional `/v1/models/{id}/load` warmup endpoint.

---

## 7. Phased roadmap

**Phase 0 — skeleton**
- Workspace scaffold, config schema + loader + validation, `/health`,
  `/v1/models` returning static config data. No inference yet.

**Phase 1 — text MVP**
- `Rune Infer-engine-llama::TextEngine` wrapping `llama-cpp-2`: load GGUF,
  single-turn and multi-turn chat via context, greedy + temp/top_p/top_k/min_p
  sampling, streaming token-by-token over SSE.
- `/v1/chat/completions` non-streaming + streaming, one model loaded at a
  time, load-on-demand.

**Phase 2 — vision**
- `VisionEngine` using `mtmd`: accept `image_url` (http(s) fetch + local
  `data:` base64), build `MtmdInputChunks`, track `total_positions()` for
  M-RoPE models correctly, generate text conditioned on image+text.

**Phase 3 — multi-model concurrency & LoRA**
- `max_loaded_models > 1`, LRU eviction, per-request LoRA application via
  `llama-cpp-2`'s LoRA adapter API, `idle_unload_seconds`.

**Phase 4 — safetensors backend**
- New `Rune Infer-engine-candle` (or similar) crate implementing the same
  `Engine` trait for non-GGUF weights, selected via `format: "safetensors"`.

**Phase 5 — new modalities**
- `Rune Infer-engine-tts`, `Rune Infer-engine-stt` (e.g. whisper.cpp bindings for
  STT — note `llama.cpp`'s ecosystem already has Voxtral/whisper mtmd audio
  support, worth reusing), `Rune Infer-engine-imagegen` (diffusion backends for
  z-image / MiniMax-H3-style models with LoRA).
- New OpenAI-compatible endpoints as needed: `/v1/audio/speech`,
  `/v1/audio/transcriptions`, `/v1/images/generations`.

**Phase 6 — config-only pipelines**
- Extend schema with a `pipelines` section: named DAGs of stages (e.g.
  `stt -> text -> tts`, or `text -> image-gen` with a LoRA swap step),
  each stage referencing a `model.id` and optional per-stage overrides
  (including LoRA choice). The `Engine` trait + `modality` dispatch from
  day one is what makes this possible without a rewrite: a pipeline
  executor just calls `Engine::generate` on each stage in order, no new
  low-level plumbing required.

---

## 8. Key crates (MVP)

| Purpose | Crate |
|---|---|
| Async runtime | `tokio` |
| HTTP server | `axum` |
| Streaming (SSE) | `axum::response::sse` |
| Serialization | `serde`, `serde_json` |
| CLI parsing | `clap` |
| LLM inference | `llama-cpp-2` (+ `llama-cpp-sys-2`), feature-gated `cuda`/`vulkan`/`metal` |
| Config hot-reload (P1) | `notify` |
| Logging | `tracing`, `tracing-subscriber` |
| Error handling | `thiserror`, `anyhow` |
| HTTP client (fetch remote image_url) | `reqwest` |

Build note: `llama-cpp-2` links C++ code, so CI and dev setup need a C/C++
toolchain (MSVC or clang on Windows, gcc/clang on Linux) and, for GPU
builds, the relevant SDK (CUDA toolkit, Vulkan SDK). Document exact
versions in README once pinned; consider prebuilt binaries per backend as
a release artifact strategy so end users don't need the toolchain.

---

## 9. Open questions to resolve during Phase 0/1

1. Multi-turn conversation state: does the server keep any session/context
   cache keyed by something, or is every request stateless (full messages
   array resent, as OpenAI does)? Recommend: stateless at the API layer
   (matches OpenAI semantics), but KV-cache prefix reuse internally as a
   performance optimization — safe to defer.
2. Concurrent requests to the *same* loaded model: llama.cpp contexts are
   generally not free-threaded for a single sequence — decide between (a)
   a single-request-at-a-time queue per loaded model, or (b) multiple
   parallel sequences via one context's batch/slot system (more complex,
   more throughput). Recommend (a) for MVP, revisit in Phase 3.
3. How strict should OpenAI compatibility be (e.g. do we need `logprobs`,
   `n>1` completions, `tools`/function-calling passthrough)? Recommend:
   MVP supports the subset real clients need (chat, streaming, vision
   content parts, basic sampling params) and documents what's unsupported
   rather than stubbing it silently.

---

## 10. Definition of done for MVP

- `cargo run --release --features cuda -- --config config/models.json`
  starts a server that:
  - Lists configured models at `/v1/models`.
  - Serves `/v1/chat/completions` (streaming + non-streaming) against at
    least one text GGUF model and one vision GGUF model (mmproj), loading
    on first request.
  - Works against an unmodified OpenAI Python/JS SDK client pointed at
    `http://localhost:8080/v1`.
  - Builds and runs on both Windows and Linux with CPU fallback when no
    GPU feature is enabled.