#!/usr/bin/env bash
# demos/tts/driver.sh
# Automates Text-to-Speech (TTS) demo request for VHS screencast recording

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/driver_helpers.sh
. "$here/../common/driver_helpers.sh"

# 1. Wait for dashboard / server to be fully responsive
server_url="http://127.0.0.1:8081/v1/audio/speech"
wait_rune_server "http://127.0.0.1:8081/health" 15

# 2. Text-to-Speech Request: Synthesize speech via Audio8 TTS Preview 0.6B
output_wav="static/artifacts/audios/tts_demo.wav"
mkdir -p "$(dirname "$output_wav")"

# Send synthesis request and write audio payload to disk.
# A failure is swallowed: the server may be shutting down.
demo_post "$server_url" '{
  "model": "audio8-tts-preview-0.6b",
  "input": "Welcome to Rune Infer. High-performance, local text-to-speech synthesis is operational.",
  "voice": "alloy",
  "response_format": "wav"
}' 30 >"$output_wav"

# 3. Hold view on dashboard to observe live inference metrics, request log, and tok/s rate
sleep 6
