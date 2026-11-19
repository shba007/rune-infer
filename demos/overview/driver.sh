#!/usr/bin/env bash

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/driver_helpers.sh
. "$here/../common/driver_helpers.sh"

server_url="http://127.0.0.1:8081/v1/chat/completions"
wait_rune_server "http://127.0.0.1:8081/health" 15

demo_post "$server_url" '{
  "model": "qwen3.8-27b",
  "messages": [
    { "role": "user", "content": "Summarize the architectural philosophy of unified AI gateways in two sentences." }
  ],
  "max_tokens": 64
}' >/dev/null

sleep 2.5

demo_post "$server_url" '{
  "model": "ternary-bonsai-2-27b-pq2-0",
  "messages": [
    { "role": "user", "content": "Explain how LRU eviction prevents CUDA out-of-memory errors." }
  ],
  "max_tokens": 64
}' >/dev/null

sleep 6
