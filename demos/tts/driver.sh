#!/usr/bin/env bash

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/driver_helpers.sh
. "$here/../common/driver_helpers.sh"

server_url="http://127.0.0.1:8081/v1/audio/speech"
wait_rune_server "http://127.0.0.1:8081/health" 15

output_wav="static/artifacts/audios/tts_demo.wav"
mkdir -p "$(dirname "$output_wav")"

demo_post "$server_url" '{
  "model": "audio8-tts-preview-0.6b",
  "input": "Welcome to Rune Infer. High-performance, local text-to-speech synthesis is operational.",
  "voice": "alloy",
  "response_format": "wav"
}' 30 >"$output_wav"

sleep 6
