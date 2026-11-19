#!/usr/bin/env bash
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$here/../../.." && pwd)"
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
fail() { echo "$1" >&2; exit 1; }

report=""
runtime=""
while [ $# -gt 0 ]; do
  case "$1" in
    --report) report="$2"; shift 2 ;;
    --runtime) runtime="$2"; shift 2 ;;
    *) fail "Unknown option '$1'" ;;
  esac
done
[ -n "$report" ] || fail '--report is required'
[ -f "$report" ] || fail "No report at $report"
report="$(cd "$(dirname "$report")" && pwd)/$(basename "$report")"
directory="$(dirname "$report")"

proof="$(node -e '
  const proof = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8").replace(/^﻿/, ""))
  const checks = Array.isArray(proof.checks) ? proof.checks : []
  if (proof.status !== "linux_cpu_parity_passed" || checks.length < 1 || checks.length > 32 || checks.some((c) => !c.passed)) {
    console.error("A completed Linux CPU parity report is required.")
    process.exit(1)
  }
  const file = String(proof.candidate?.file ?? "")
  if (!file || file !== require("path").win32.basename(file) || file !== require("path").posix.basename(file)) {
    console.error("The candidate must be a file beside the report.")
    process.exit(1)
  }
  console.log([file, proof.candidate.sha256, checks.length].join("\t"))
' "$(native "$report")")" || exit 1
IFS=$'\t' read -r candidate_file candidate_sha check_count <<<"$proof"
candidate="$directory/$candidate_file"
[ -f "$candidate" ] || fail "No candidate at $candidate"
[ -n "$runtime" ] || runtime="$repository/bin/litert/libLiteRt.dll"
[ -f "$runtime" ] || fail "No runtime at $runtime"
runtime="$(cd "$(dirname "$runtime")" && pwd)/$(basename "$runtime")"

cd "$repository" || exit 1
build="$(cargo build --locked --lib --target-dir target/ci-check --message-format=json)" ||
  fail 'Infer library build failed.'
rlibs="$(printf '%s\n' "$build" | node -e '
  const lines = require("fs").readFileSync(0, "utf8").split(/\r?\n/).filter((l) => l.startsWith("{"))
  const artifacts = lines.map((l) => JSON.parse(l)).filter((m) => m.reason === "compiler-artifact")
  const rlib = (name) => artifacts.filter((a) => a.target.name === name).flatMap((a) => a.filenames).filter((f) => f.endsWith(".rlib"))
  const anyhow = rlib("anyhow"), sha2 = rlib("sha2")
  if (anyhow.length !== 1 || sha2.length !== 1) {
    console.error("Ambiguous validation dependencies.")
    process.exit(1)
  }
  console.log(anyhow[0] + "\t" + sha2[0])
')" || exit 1
IFS=$'\t' read -r anyhow sha2 <<<"$rlibs"

rustc --edition 2024 -D warnings "$(native "$here/candidate_worker.rs")" \
  -L dependency=target/ci-check/debug/deps --extern "sha2=$sha2" \
  --extern "anyhow=$anyhow" -o target/depth-large-candidate-worker.exe ||
  fail 'Candidate worker build failed.'
rustc --edition 2024 -D warnings "$(native "$here/native_check.rs")" -o target/depth-large-native-check.exe ||
  fail 'Native checker build failed.'
worker="$(native "$repository/target/depth-large-candidate-worker.exe")"

index=0
while [ "$index" -lt "$check_count" ]; do
  RUNE_LITERT_CANDIDATE_SHA256="$candidate_sha" target/depth-large-native-check.exe \
    "$worker" "$(native "$runtime")" "$(native "$candidate")" "$(native "$directory/fixture-$index")" ||
    fail "Windows CPU parity failed for fixture $index."
  index=$((index + 1))
done
