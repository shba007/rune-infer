#!/usr/bin/env bash
# demos/overview/driver.sh
# Automates demo requests for VHS screencast recording (Dashboard, Lazy-load, and LRU swap)

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/driver_helpers.sh
. "$here/../common/driver_helpers.sh"

# 1. Wait for dashboard / server to be fully responsive
server_url="http://127.0.0.1:8081/v1/chat/completions"
wait_rune_server "http://127.0.0.1:8081/health" 15

# 2. First Request: Lazy-load Qwen3.8 27B
demo_post "$server_url" '{
  "model": "qwen3.8-27b",
  "messages": [
    { "role": "user", "content": "Summarize the architectural philosophy of unified AI gateways in two sentences." }
  ],
  "max_tokens": 64
}' >/dev/null

# 3. Short 2.5s pause to observe completion, tok/s rate, and latency sparkline
sleep 2.5

# 4. Second Request: Ternary Bonsai 27B (triggers LRU model swap)
demo_post "$server_url" '{
  "model": "ternary-bonsai-2-27b-pq2-0",
  "messages": [
    { "role": "user", "content": "Explain how LRU eviction prevents CUDA out-of-memory errors." }
  ],
  "max_tokens": 64
}' >/dev/null

# 5. Hold final state
sleep 6
