#!/usr/bin/env bash

wait_rune_server() {
  local url="${1:-http://127.0.0.1:8081/health}" timeout="${2:-20}" waited=0
  while [ "$waited" -lt $((timeout * 4)) ]; do
    if curl --silent --fail --max-time 2 --output /dev/null "$url"; then return 0; fi
    sleep 0.25
    waited=$((waited + 1))
  done
  return 1
}

demo_post() {
  local url="$1" payload="$2" timeout="${3:-90}" api_key="${4:-rune-demo-secure-key}"
  curl --silent --fail --max-time "$timeout" \
    --header "Authorization: Bearer $api_key" --header 'Content-Type: application/json' \
    --data "$payload" "$url" || true
}
