# Studio integration contract

This guide describes the current source contract. It does not establish a released
minimum version, successful inference for every backend, or measured GPU latency.

## Scheduling and attribution

Background requests require a resident model and a cancellation-capable engine.
Foreground activity refuses new background work with HTTP 429, `Retry-After: 1`,
error type `server_busy`, code `busy`. Cold background work returns HTTP 503
`background_requires_resident`; unsupported engines or cloud chat return HTTP
503 `background_unsupported`. Llama-server and llama-server-prism implement
cancellation; real slot-release latency remains unmeasured.

Running background work yields with HTTP 503 and code `yielded`. An open SSE
response retains HTTP 200 and sends an `error` event with code `yielded`.
Clients must inspect SSE events. `x-no-swap` checks residency and does not grant
locality authorization.

Model execution routes expose `x-model-served`, `x-model-locality` and
`x-model-resident`, with loading/eviction metadata where applicable. Composite
audio responses attribute the primary model only. Image variations currently
echo the input without running a model and have no model attribution. Tests
inject engines; they do not prove all modalities work with real models.

## Audio, decisions and grounding

Audio uploads are limited to 25 MiB per file (HTTP 413 `file_too_large`).
Transcription selects the first SpeechToText catalog entry when `model` is absent.
Backend-specific duration limits and latency remain unmeasured. The response has
required `text` and optional `language`, `duration`, `segments`, `words`,
`speakers` and `speaker_activity`; it has no top-level confidence field.
Speech models share registry/LRU limits with chat.

Decision probabilities are not empirically calibrated. Encoder grounding is
unsupported; llama grounding rejects missing, malformed, reversed, degenerate or
out-of-range boxes instead of inventing a center. Grounding boxes use
`[ymin, xmin, ymax, xmax]` normalized to 0..1000. A numeric confidence of 0.0 does
not establish calibration. The off-center PNG and expected JSON under
`tests/fixtures/studio/` are required regression fixtures.

Run the optional live visual acceptance probe explicitly:

```powershell
$env:RUNE_VISUAL_BASE_URL = 'http://127.0.0.1:8080'
$env:RUNE_VISUAL_MODEL = '<local vision model id>'
# Set RUNE_VISUAL_API_KEY if the server requires authentication.
cargo test --test studio_visual_fixture_tests -- --ignored --nocapture
```

## Residency, telemetry and acceptance gaps

`server.idle_unload_seconds` is global, defaults to 300, and is overridden by
`--idle-timeout`. Zero disables idle eviction but not capacity/LRU eviction.
Health/models queries do not pin models. Unknown VRAM figures are omitted rather
than serialized as null or invented zero. Device usage is not per-process usage.
Co-residency, real model quality, GPU timing and other-platform behavior require
separate hardware acceptance checks.

For synthetic TUI frame export, see the [review harness](../scripts/tui-review/README.md).
Known review gaps include clipping at narrow widths, lack of a distinct in-flight
busy state, idle dots without status words, and incomplete font coverage in PNGs.
`NO_COLOR` and `RUNE_ASCII=1` are tested as full-frame fallbacks.

## Development checks

```text
cargo test --locked
cargo test --no-default-features --lib --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --check
cargo build --locked
```

Live/hardware tests remain ignored by default. Test names and assertions are the
maintained evidence; historical milestone narratives are not current guarantees.
