# Screen Insight integration

Studio captures the screenshot and orchestrates the one-shot workflow. Infer serves
the existing HTTP image-input API; no workflow engine or MCP transport is added.

Start with `rune-infer serve`: the existing dashboard opens on a TTY. Use
`rune-infer serve --headless` without a terminal. Bare `rune-infer` remains compatible.
`--config`, `RUNE_HOME`, development-checkout precedence and ~/rune/infer migration
are unchanged. No binary/storage names changed.

## Tested request

POST /v1/chat/completions with Content-Type application/json and Authorization:
Bearer YOUR_LOCAL_KEY when configured:

```json
{
  "model": "minicpm-o-4-5-q8-0",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe the colored shapes and their positions in one sentence."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_PNG_BYTES"}}
  ]}],
  "thinking": false,
  "reasoning_effort": "none",
  "temperature": 0,
  "max_tokens": 64
}
```

Use the exact configured ID, not an assumed family name. The tested local catalog
entry is MiniCPM-o 4.5 Q8_0 with llama-server, its vision-F16 GGUF projector in
extra.mmproj_path, 4096 context and max_tokens 64. Existing weights and the upstream
llama-server executable were reused; no model was downloaded. Vision/image input
must be present in the configured capabilities and a usable matching projector
must exist. Text-only entries must not claim vision support.

Windows CUDA, Infer 0.8.0: the repository's off-center PNG returned HTTP 200 with
chat completion content: “A red square is positioned in the upper right corner of
the image.” x-model-served matched the selected ID and x-model-locality was local.
This proves one real image path, not general screenshot accuracy, grounding quality
or audio support. A separate client completed the same image request after the
first client's SSE connection was closed after receiving initial data. Health also
remained HTTP 200. Backend cancellation/slot-release latency was not measured;
disconnecting does not guarantee immediate GPU work termination for every engine.

Evidence is retained locally in target/screen-insight-live.json. Isolated migrated
homes preserved original config/environment, Kit sentinels and external weights.
An initial Windows trial failed when a rebased relative model path became too long
for the upstream engine. The successful trial retained the original absolute model
reference. Review long model paths on Windows; no silent model fallback was added.

## Refusals and client behavior

- Missing vision capability: 400 model_vision_unsupported; missing projector:
  model_projector_missing. These preflight paths have source regression tests;
  the live trial did not alter capabilities to force those errors.
- Unknown selected model: live GET /v1/models/ID returned 404 model_not_found.
  Runtime load failures can also surface as errors; do not silently choose another ID.
- Background scheduling: 429 busy with Retry-After: 1; cold background models:
  503 background_requires_resident; unsupported cancellation backends/cloud:
  503 background_unsupported. Interrupted background work: 503 yielded, or an SSE
  error event after HTTP 200. Inspect stream events, not only HTTP status.
- x-no-swap can return 409 swap_refused instead of loading a cold model. It does
  not enforce locality authorization. Inspect configured locality before sending
  sensitive images and response attribution afterward.

No demonstrated API blocker remains for the tested one-shot image request.
Native Linux/macOS/ARM, actual Studio UI capture, broader model quality and
simultaneous-client GPU latency remain unverified. See STUDIO-CONTRACT.md.

## Future planning only

| Workflow | Existing support and gaps |
| --- | --- |
| Recorded transcription/subtitles | Multipart /v1/audio/transcriptions; text required, segments/words/duration optional and backend dependent. Confirm real timestamps before generating subtitles. |
| Streaming transcription | No established live incremental transcription transport; do not treat chat SSE as streaming ASR. |
| Speaker diarization | Separate /v1/audio/diarizations API/capability; requires an actually integrated/configured engine. A named Nemotron model alone does not establish support. |
| Image comparison | Chat accepts multiple image_url parts with an explicitly capable model; real comparison quality untested here. |
| Frame-based video | Existing media/frame input handling; Studio must choose frames and timestamps. No workflow scheduler or continuous video guarantee. |

These are plans and acceptance gaps, not new implementations.
