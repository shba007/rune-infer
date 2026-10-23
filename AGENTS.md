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
  checkout > `~/rune/infer`). The process `chdir`s into the root so the
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
- `detect.rs` — what a GGUF file actually *is*, read from its header (architecture,
  distinctive keys such as `audiocpp.model_spec.family` / `prism.*`, and tensor names)
  rather than its file name: chat / vision chat, image generation, text-to-speech,
  speech-to-text. It pairs vision projectors by size (`clip.vision.projection_dim`
  must equal the model's `embedding_length`), attaches MTP draft heads, recognises
  metadata-less diffusion files by tensor names, and reports every file it skipped
  and why. Used by the setup wizard.
- `schema.rs` — the on-disk schema 2 types (`ModelEntry`, `LocalBlock`,
  `RemoteBlock`), per-engine `extra` validation (`allowed_extra_keys`) and the
  conversion into the flat `ModelConfig` the rest of the server uses.
- `migrate.rs` — `rune-infer migrate` (schema 1 → 2); the only place that still
  infers an engine from model names/architectures.
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
- **Studio priority contract:** Background work requires a resident model and a cancellation-capable engine; uncancellable engines/cloud chat refuse it with 503. Foreground arrival yields running background work (503, or an SSE error after HTTP 200). Full-frame `NO_COLOR` and `RUNE_ASCII=1` fallbacks are supported. See `docs/STUDIO-CONTRACT.md` for the contract and remaining hardware/UX gaps.
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

- The wizard cannot find an image model's companion files (VAE, text encoder,
  `llm_vision_path`): they are separate downloads in other repos. It flags those
  models, leaves them unticked and says which `extra` keys to add.
- Detection of language-model fine-tunes for speech ("…TTS", "…ASR") relies on
  whole-word matches in the name; verify them. Vevo2 is set up as text-to-speech
  although it also does voice conversion.
- **Capabilities come from evidence, never from a family name.** Omni families
  (Gemma-4, MiniCPM-o, `…Omni…`) are recognised by architecture or file name
  (MiniCPM-o's body is a plain `qwen3`, `general.name` "Llm") but that is only a
  hint: the wizard starts them as chat models and sets modality and capabilities
  from the projector it pairs, using that file's own `clip.has_vision_encoder` /
  `clip.has_audio_encoder` flags — both → `Omni`, vision → `VisionText`. A
  projector belongs to the models in the nearest enclosing folder that holds a
  model (`snapshot/vision/x.gguf` → `snapshot/model.gguf`); `llama-server` loads a
  single `--mmproj`, so the most capable match is attached (vision wins ties) and
  the rest are noted. A separate audio-encoder file (MiniCPM-o's `audio/`, arch
  `whisper` + `audio_projector`) is a companion, not a speech model. A catalog
  entry must likewise only claim what its files provide: the dashboard's
  `C V T A I` flags just read `capabilities`.
- **The server refuses media a model cannot take**, before loading it: audio to a
  model without audio support → `model_audio_unsupported`; image/audio to a
  llama.cpp model whose `extra.mmproj_path` is missing or points at no file →
  `model_projector_missing`. (Previously both reached `llama-server` and came
  back as a bare 500.)
- Draft heads (MTP / `…-assistant` architectures) are never listed as models.
- `rune-infer setup` skips models whose id already exists, so re-running it does
  not repair entries written by an older, less accurate wizard.

Known divergences from the idealised design: the trait is synchronous and
narrow (`execute` with an optional token callback); `Modality` dispatch is
partly a `match` in `registry.rs`; ``notify` is a dependency but config
hot-reload is **not implemented yet**.

---

## 4. Config

`<home>/config/` (home = `~/rune/infer`, see §3) uses **schema 3**, decoupled
into `server.json` (host, port, security limits) and `catalog.json` (model entries).
Adding a model is a `catalog.json` edit only — no source change and no server
restart needed (runtime hot-reload detects changes every 2s).

`config/server.json`:
```json
{
  "schema_version": 3,
  "server": {
    "host": "127.0.0.1", "port": 8080, "api_key": "{{RUNE_API_KEY}}",
    "max_loaded_models": 1, "idle_unload_seconds": 300, "vram_budget_ratio": 0.95
  }
}
```

`config/catalog.json`:
```json
{
  "schema_version": 3,
  "models": [
    {
      "id": "qwen3.8-27b",
      "name": "Qwen3.8 27B",
      "description": "Multimodal reasoning model",
      "modality": "VisionText",
      "capabilities": ["Chat", "Vision", "Tool Call"],
      "engine": { "supported": ["llama-server"], "preferred": "llama-server" },
      "max_context_length": 262144,
      "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 40, "max_tokens": 4096 },
      "local": {
        "path": "D:/Models/qwen3.8-27b.gguf",
        "params": 27000000000, "bits_per_weight": 3.44,
        "context_length": -1, "gpu_layers": 99, "args": "-fa on"
      },
      "extra": { "mmproj_path": "D:/Models/mmproj-BF16.gguf" }
    },
    {
      "id": "gemini-3.7-flash",
      "name": "Gemini 3.7 Flash",
      "description": "",
      "modality": "VisionText",
      "capabilities": ["Chat", "Vision"],
      "max_context_length": 1048576,
      "sampling": {},
      "remote": {
        "provider": "google", "upstream_model": "gemini-3.7-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "{{GEMINI_API_KEY}}",
        "rate_limit": { "requests_per_minute": 5, "max_concurrent": 2, "retry_delay_ms": 2000 }
      },
      "extra": {}
    }
  ]
}
```

- **Core keys** (every model): `id`, `name`, `description`, `modality`,
  `capabilities`, `max_context_length`, `sampling`, `extra`. Each model also has
  exactly one of `local` (own weights) or `remote` (cloud provider). Local
  models additionally require `engine`; cloud models must not have one (the
  `remote.provider` adapter handles them). The only keys that differ between
  models are inside `extra`.
- **`engine`**: `{ "supported": [...], "preferred": "..." }`. `preferred`
  (default: first supported) is tried first; if it fails to start, the others
  are tried in the listed order. Names: `llama-server`, `llama-server-prism`,
  `sd-server`, `audio-cpp`, `nemo-speech`, `crispasr`, `encoder`, `needle`.
  Nothing is inferred from names or architectures any more. The engine that
  started is shown in the dashboard and the load event.
- **`local`**: `path`, `params`, `bits_per_weight` (the last two only feed the
  VRAM estimate), `context_length` (`-1` = automatic), `gpu_layers`, `args`
  (extra command-line arguments passed verbatim to the engine process).
- **`remote`**: `provider` (`openai` | `anthropic` | `google` | `openrouter` |
  `custom`), `upstream_model`, `api_key`, `base_url`, `rate_limit`
  (requests/tokens per minute and day, concurrency, queueing, retries; a single
  `retry_delay_ms`).
- **`extra`** holds engine-specific settings and is validated against the
  model's `supported` engines when the config loads (a key no engine uses is an
  error that lists the allowed keys): `llama-server*`: `mmproj_path`,
  `mtp_path`, `mtp_heads`, `max_resolution`; `sd-server`: `vae_path`,
  `text_encoder_path`, `llm_vision_path`, `default_steps`, `default_cfg_scale`,
  `default_sample_method`, `max_resolution`, `lora` (a list of
  `{name, path, scale}` — parsed and validated but **not yet applied** by any
  engine); other engines take none yet. To support a new engine setting, add
  the key to `allowed_extra_keys` in `schema.rs`.
- **Strict parsing:** unknown keys anywhere in a model are errors, so a typo
  can no longer be silently ignored.
- **No hardcoded model names in source.** `find()` resolves an exact id, then a
  case-insensitive id or display name; unknown generic OpenAI-SDK default names
  (`text-embedding-3-small`, `whisper-1`, `tts-1`, …) fall back to the first
  model of that modality. Anything else is "model not found".
- **Catalog order is priority.** When a request omits `model`, the audio,
  embedding, moderation, NLU, OCR and detection handlers use the *first*
  catalog entry of the right modality, so reordering `catalog.json` can change
  defaults. The shipped catalog is ordered local before cloud, then by modality
  (chat → image → speech → analysis/utility), then size (largest first; cloud by
  context window), then quality (higher `bits_per_weight` first), then id.
- **Upgrading from schema 1 & 2:** `rune-infer migrate [--dry-run]` splits the
  unified `models.json` into `config/server.json` and `config/catalog.json`, keeps
  a `models.json.v2.bak` backup, and validates the result before writing. A
  legacy configuration is refused at startup with a pointer to this command.
- **Runtime Hot-Reload:** `server::serve_with_paths` polls filesystem metadata
  every 2 seconds. When `catalog.json` or `server.json` is modified, the router
  reloads without interrupting in-flight requests or restarting the process.

Rules:
- **Secrets never go inline.** Use `"api_key": "{{GEMINI_API_KEY}}"` style
  references resolved from `.env`/system env. `.env` is gitignored; the
  pre-commit hook scans for secrets. The same templating works for any string,
  e.g. a shared `{{MODELS_DIR}}` prefix in model paths.
- The config is expected to **split into multiple files** over time (server
  settings, model catalog, secrets). Bump `schema_version` and add a `migrate`
  step for any shape change.
- Env vars used today: `RUNE_HOST`, `RUNE_PORT`, `RUNE_API_KEY`,
  `LLAMA_SERVER_PATH`, `SD_SERVER_PATH`, `CRISPASR_PATH`, `NEMO_SPEECH_PATH`,
  `AUDIO8_PATH`, `S3_*`/`AWS_*`, `RUSTFS_*`, `GH_TOKEN`, provider keys.

---

## 5. API surface

OpenAI canonical: chat completions (stream + non-stream, vision, video, tools,
`response_format` JSON/schema), embeddings, moderations, images
(generations/edits/variations), audio (transcriptions/translations/speech),
models, health. (Note: `/v1/responses` and Anthropic Messages at `/v1/messages`
and `/anthropic/v1/messages` are implemented but currently disabled in schema 2
routing pending full protocol alignment). Rune-specific: `/v1/inference`,
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
  optional local model discovery (each GGUF is classified from its header; projectors and drafts are attached; incomplete image models start unticked) →
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
  (abort on mismatch), install to the shared `~/rune` directory on every platform, and handle upgrades. Env overrides:
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

- Implement and test locally; leave changes uncommitted. Do not stage, commit, push, merge, create tags, publish releases or dispatch publishing workflows. The owner performs those actions; this overrides older handoffs.
- Prepare the version and its matching CHANGELOG section before the owner commits. New public descriptions cover user-visible behavior and migration, without internal planning identifiers. Preview with `python scripts/release_notes.py`; never regenerate release notes from Git history.
- CI is read-only: it must not auto-commit formatting, bump versions, or push release branches. After owner promotion to main and an owner-created version tag push, release automation may publish that existing tag and its assets, using the exact changelog section. No automation creates source commits or pushes branches/tags.
- Shared executables live directly in `<user-home>/rune`; Infer owns only its executable and `infer/` home. Do not modify Kit, `tools/`, plugin payloads, unrelated projects or model weights. `RUNE_HOME` remains Infer-specific. Default home migration preserves originals and refuses conflicts; see `docs/INSTALLATION.md`.

- **Ask before adding new features or new dependencies/libraries.**
- Targets: Windows + Linux + macOS. TUI stack is `ratatui` + `crossterm` +
  `tui-input` (approved); anything else new needs approval first.
- Conventional commits (enforced by `.cargo-husky` hooks; changelog via
  `git-cliff`/`cog`). Edition 2024, `rustfmt.toml` respected.
- Audit log goes to `logs/rune-infer.log`; payload bodies are never logged.
- Build needs a C/C++ toolchain only if engines are built from source; FFmpeg
  must be on `PATH` for video input.
