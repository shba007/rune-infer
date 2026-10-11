# Rune Infer — Project Guide

Rune Infer is a standalone, config-driven **unified AI gateway** written in Rust,
shipped as both a **terminal app** (live dashboard + setup wizard) and a
**library** (`rune_infer::server::serve` embeds the HTTP server; the `tui`
cargo feature is on by default and can be dropped with
`default-features = false`).
It puts local models (GGUF, safetensors, cact) and cloud providers (Gemini,
OpenAI-compatible, Anthropic, OpenRouter) behind one HTTP endpoint, selected
per request by the `model` id.

**Success for 1.0:** every AI call the owner makes in a day — any modality,
local or cloud — goes through this one endpoint. It is intended as the
foundation of a larger Rune ecosystem.

This document is the source of truth for scope, architecture and direction.
Update it whenever a decision below changes. (Rewritten 2026-10-11 to match the
code as built; the original plan assumed in-process `llama-cpp-2` FFI and a
Cargo workspace, neither of which is how the project turned out.)

---

## 1. Philosophy

1. **Breadth is the product.** One server for LLM, vision, video, STT, TTS,
   image generation/editing, OCR/OMR, NLU, moderation, detection and
   embeddings. New models/modalities are welcome even if rough.
2. **Never crash the host.** Pre-flight context/token/VRAM budgeting, engines
   isolated in subprocesses, reject a request rather than OOM or `SIGABRT`.
3. **Performance matters**, but never at the cost of rule 2. Per-model request
   concurrency is *not* a priority yet.
4. **Config over code.** Adding a model should be a `models.json` edit. A small
   Rust engine per genuinely novel architecture is acceptable where no upstream
   tool exists (e.g. `needle`, `nemo_speech`, `encoder`).
5. **Orchestrate, don't reinvent.** Prefer driving best-in-class upstream
   engines (`llama-server`, `sd-server`, `crispasr`) as managed subprocesses;
   write native Rust only where that is the practical option.
6. **Explicit routing.** The caller names the exact model id. No aliases,
   fallback chains or capability-based routing (revisit only if asked).
7. **Pragmatic-superset compatibility.** Unmodified OpenAI and Anthropic SDKs
   must work. Rune extensions (extra fields, `/v1/inference`) are fine; document
   unsupported parameters rather than stubbing them silently.

---

## 2. Scope

### In scope
- Local inference via managed engine subprocesses + native Rust engines.
- Cloud proxying with rate limiting, concurrency caps, queueing and retries
  (`src/proxy/`), so cloud and local models are interchangeable to clients.
- Auto-installing/downloading **engine binaries** (`llama_server/install.rs`,
  `engines/download.rs`).
- Model weight **`pull`** (planned — not yet implemented).
- Per-request LoRA swapping, multi-model concurrency with VRAM-budgeted LRU
  and idle unload.
- Gateway features: usage/cost tracking, metrics, caching/dedupe, remote-model
  rate limits and retries.

### Non-goals
- **No agent/orchestration loop.** The server exposes inference primitives;
  callers compose stages (stt → llm → tts, etc.) and drive tool loops. No
  config-defined pipelines/DAGs. (`/v1/audio/speech-to-speech` exists as a
  legacy convenience; do not extend that pattern.)
- Not a chat UI. Not a training/fine-tuning tool. Not a distributed server —
  single process, single machine.
- Not hardened for multi-tenant public internet in the near term (single API
  key today; per-key quotas are not planned yet).

---

## 3. Architecture (as built)

Single crate `rune-infer` (`src/lib.rs` + `src/main.rs`), axum HTTP server.

```
 clap CLI (main.rs) ── dotenvy ── config.rs (models.json, {{ENV}} templating)
        │
   axum router (api.rs) ── auth middleware (Bearer / x-api-key)
        │   /v1/chat/completions  /v1/responses  /v1/messages (+/anthropic/*)
        │   /v1/images/*  /v1/audio/*  /v1/embeddings  /v1/moderations
        │   /v1/nlu/intents  /v1/ocr  /v1/inference  /v1/models  /health
        │
   ┌────┴────────────────────────┐
   │ local model                 │ remote model
   │ inference/registry.rs       │ proxy/ (controller + adapters:
   │ lazy load, VRAM budget,     │  openai, anthropic, gemini)
   │ LRU eviction                │ rate limit, queue, retries
   └────┬────────────────────────┘
        │ InferenceEngine trait (inference/traits.rs)
   engines/: llama_server (subprocess, auto-install), sd_server, crispasr,
             nemo_speech, audio, needle (cact), encoder (safetensors BERT-style)
```

Key modules:
- `home.rs` — workspace resolution (`RUNE_HOME` > `./config/models.json` dev
  checkout > `~/.rune-infer`). The process `chdir`s into the root so the
  engines' relative `bin/`, `logs/`, `config/` and `.env` just work.
- `server.rs` — `bind()` (actionable port errors) and `serve()` with graceful
  shutdown; used by the CLI and by library embedders.
- `telemetry.rs` — `tracing` layer converting audit/lifecycle events into typed
  `UiEvent`s (`RequestRecord`, `ModelEvent`, `LogLine`). Producers call
  `record_completion/record_failure/model_loading/...`; nothing else needs to
  know about the UI.
- `setup.rs` — UI-agnostic first-run logic: prerequisite detection, GGUF header
  sniffing, local model discovery, cloud provider presets + key validation,
  config/`.env` merge-writing.
- `tui/` (feature `tui`) — ratatui + crossterm. `dashboard.rs` (models with
  load state, VRAM gauge, tok/s + latency sparklines, recent requests, logs),
  `wizard.rs` (setup), `theme.rs`.
- `config.rs` — `ModelRegistry`, `ModelConfig`, `Modality`, `ProviderType`,
  validation, capability resolution, token budgets. `{{VAR}}` placeholders are
  resolved from the environment (`.env` via dotenvy) at load time.
- `inference/registry.rs` — engine creation, GPU VRAM detection, context sizing,
  LRU eviction bounded by `max_loaded_models` and `vram_budget_ratio`.
- `inference/process.rs` — subprocess lifecycle (Windows job objects / unix
  libc) so children die with the server.
- `api/` — one handler module per endpoint family; `api/chat/` splits prompt,
  tools, stream and response handling. `media.rs` handles image/video input
  (URL, data URI, local path, S3-compatible streaming) and uses FFmpeg.
- `proxy/` — cloud adapters translating the internal chat request to each
  provider.

Known issues worth fixing:
- `cuda`/`vulkan`/`metal` cargo features are empty; engines are fetched at
  runtime by host detection, so one binary per OS suffices.
- Local engines swallow load progress; the dashboard shows only
  loading → loaded/failed with duration and VRAM delta.
- Throughput is measured around `engine.execute` (prompt processing + decode),
  not decode-only.

Behaviour notes:
- **Sampling** (`temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`) is
  layered: request value > model `sampling` block > engine default. It applies
  to local chat/responses and to remote providers (`top_k`/`min_p` are only
  forwarded when set — avoid them on providers that reject unknown params).
  `max_tokens` from config also becomes the reply reserve in the context
  pre-flight, so a low value can truncate long reasoning output.
- **Idle unload**: `server.idle_unload_seconds` (default 300, `0` = never;
  `--idle-timeout` overrides) unloads a model with no requests for that long.
  A model referenced by an in-flight request is never unloaded. The next
  request pays a cold load. `max_loaded_models` of `0` means unlimited
  (`--max-loaded-models` overrides).

Known divergences from the idealised design: the trait is synchronous and
narrow (`execute` with an optional token callback); `Modality` dispatch is
partly a `match` in `registry.rs`; ``notify` is a dependency but config
hot-reload is **not implemented yet**.

---

## 4. Config

`<home>/config/models.json` (home = `~/.rune-infer`, see §3; `schema_version: 1`) — `server` block (host, port,
`api_key`, `max_loaded_models`, `idle_unload_seconds`, `vram_budget_ratio`) and
a `models` array. Per model: `id`, `name`, `modality`, `architecture`,
`format` (`gguf` | `safetensors` | `cact`), `provider` (`local` default, or
`openai`/`anthropic`/`google`/`openrouter`/`custom` with `base_url`,
`upstream_model`, `api_key`, `rate_limit`), paths (`model_path`, `mmproj_path`,
`vae_path`, `text_encoder_path`, `llm_vision_path`, `mtp_path`), image defaults,
`capabilities`, size metadata used for budgeting, `sampling`, `runtime`, `lora`.

Rules:
- **Secrets never go inline.** Use `"api_key": "{{GEMINI_API_KEY}}"` style
  references resolved from `.env`/system env. `.env` is gitignored; the
  pre-commit hook scans for secrets.
- The config is expected to **split into multiple files** over time (server
  settings, model catalog, secrets). Keep schema changes versioned via
  `schema_version` with a migration function.
- Env vars used today: `RUNE_HOST`, `RUNE_PORT`, `RUNE_API_KEY`,
  `LLAMA_SERVER_PATH`, `SD_SERVER_PATH`, `CRISPASR_PATH`, `NEMO_SPEECH_PATH`,
  `AUDIO8_PATH`, `S3_*`/`AWS_*`, `RUSTFS_*`, `GH_TOKEN`, provider keys.

---

## 5. API surface

OpenAI canonical: chat completions (stream + non-stream, vision, video, tools,
`response_format` JSON/schema), responses, embeddings, moderations, images
(generations/edits/variations), audio (transcriptions/translations/speech),
models, health. Anthropic Messages at `/v1/messages` and `/anthropic/v1/messages`
so Anthropic SDK clients can target local models. Rune-specific: `/v1/inference`,
`/v1/nlu/intents`, `/v1/ocr`, `/v1/images/{captions,detections,embeddings,
recognize,upscales,restorations,style-transfers}`, `/v1/audio/speech-to-speech`.

Errors mirror OpenAI's `{"error": {...}}` shape. Postman collection in
`postman/` documents the surface.

---

## 6. Terminal UX & install

- `rune-infer` on a TTY opens the dashboard with the server running inside it;
  `--headless` (or no TTY: Docker, systemd, CI) runs a plain server with console
  logs. With no config present it first runs the **setup wizard** (also
  `rune-infer setup` any time): welcome/system check → network & API key →
  optional cloud keys (validated, stored in `.env`, config holds `{{VAR}}`) →
  optional local GGUF discovery (architecture + mmproj auto-detected) →
  optional `llama-server` pre-download → review → write. Nothing is written
  before the review; every step is skippable.
- Secure defaults: wizard binds `127.0.0.1`; choosing the network requires an
  API key; the dashboard shows a red banner when listening beyond localhost
  without a key.
- While the TUI is active nothing may write to stdout/stderr: use `tracing`,
  never `println!`.
- Releases publish version-less archives per platform
  (`rune-infer-{linux-x64,linux-arm64,macos-x64,macos-arm64,windows-x64,windows-arm64}.{tar.gz,zip}` +
  `.sha256`) so `releases/latest/download/<name>` works without API calls.
  `install.sh` / `install.ps1` detect the platform, verify the checksum
  (abort on mismatch), install to `~/.local/bin` or
  `%LOCALAPPDATA%\Programs\rune-infer`, and handle upgrades. Env overrides:
  `RUNE_VERSION`, `RUNE_INSTALL_DIR`, `RUNE_BASE_URL`, `RUNE_NO_MODIFY_PATH`.
- Known gaps: release binaries are unsigned (SmartScreen/Gatekeeper warn); no
  uninstall script. `windows-arm64` is built `continue-on-error` (newest
  target; `install.ps1` falls back to the x64 build under emulation if it is
  missing). arm64 builds use GitHub's native arm runners, which are free only
  for public repositories. The runtime engine downloaders are arch-aware on
  the main path, but several offline fallbacks (llama-server, sd-server,
  crispasr, nemo-speech) still assume x64 on Linux/Windows - verify before
  promising arm64 engine support.

---

## 7. Roadmap (priority order)

1. **Stability & test hardening** — currently one integration-test file
   (5 tests). Cover config parsing/templating, budgeting, each adapter,
   handlers; keep CI (`integrate.yml`) green.
2. **More modalities/models** — video, music, segmentation, more TTS/STT, as
   upstream engines allow.
3. **Gateway features** — usage/cost tracking and metrics, caching/request
   dedupe; extend existing remote rate limiting and retries.
4. **Packaging & distribution** — *installers, checksummed archives and the
   wizard are done*; remaining: code signing, arm64 builds, Docker images,
   model `pull`, cloud/container deployment.
5. **Per-request LoRA** and **multi-model concurrency + idle unload**
   (`max_loaded_models > 1`; LRU exists, idle unload and LoRA do not).
6. **Config hot-reload** and config file splitting.

Deferred: per-model request concurrency / parallel slots.

---

## 8. Working agreements

- **Ask before adding new features or new dependencies/libraries.**
- Targets: Windows + Linux + macOS. TUI stack is `ratatui` + `crossterm` +
  `tui-input` (approved); anything else new needs approval first.
- Conventional commits (enforced by `.cargo-husky` hooks; changelog via
  `git-cliff`/`cog`). Edition 2024, `rustfmt.toml` respected.
- Audit log goes to `logs/rune-infer.log`; payload bodies are never logged.
- Build needs a C/C++ toolchain only if engines are built from source; FFmpeg
  must be on `PATH` for video input.
