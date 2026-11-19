#!/usr/bin/env bash
set -uo pipefail

native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
absolute() { printf '%s/%s' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
fail() { echo "$1" >&2; exit 1; }

probe_directory="" worker="" checker="" runtime="" ir_graph=false
while [ $# -gt 0 ]; do
  case "$1" in
    --probe-directory) probe_directory="$2"; shift 2 ;;
    --worker) worker="$2"; shift 2 ;;
    --checker) checker="$2"; shift 2 ;;
    --runtime) runtime="$2"; shift 2 ;;
    --ir-graph) ir_graph=true; shift ;;
    *) fail "Unknown option '$1'" ;;
  esac
done
for required in probe_directory worker checker runtime; do
  [ -n "${!required}" ] || fail "--${required//_/-} is required"
done
[ -d "$probe_directory" ] || fail "No probe directory at $probe_directory"
for file in "$worker" "$checker" "$runtime" "$probe_directory/probes.json"; do
  [ -f "$file" ] || fail "No file at $file"
done
directory="$(cd "$probe_directory" && pwd)"
worker="$(absolute "$worker")"
checker="$(absolute "$checker")"
runtime="$(absolute "$runtime")"

# One line per probe: <operator> <model sha256> <reference sha256> <shape, comma separated>
manifest="$(node -e '
  const proof = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8").replace(/^﻿/, ""))
  const probes = Array.isArray(proof.probes) ? proof.probes : []
  const bad = (message) => { console.error(message); process.exit(1) }
  if (proof.source_sha256 !== "6f03564f13abfae18185581a809cb5c0ee5c0a425dc5f85102194b2b321eba3d" || probes.length < 1 || probes.length > 32)
    bad("Invalid diagnostic manifest.")
  console.log(String(proof.input_sha256))
  for (const probe of probes) {
    if (!/^\d+$/.test(String(probe.operator)) || !/^[a-f0-9]{64}$/.test(String(probe.model_sha256))) bad("Invalid probe identity.")
    console.log([probe.operator, probe.model_sha256, probe.reference_sha256, [probe.shape].flat().join(",")].join("\t"))
  }
' "$(native "$directory/probes.json")")" || exit 1

output="$(mktemp -d "$directory/windows-XXXXXXXXXXXX")"
results="$output/results.tsv"
: >"$results"
input_sha="$(printf '%s\n' "$manifest" | head -n 1)"

while IFS=$'\t' read -r operator model_sha reference_sha shape; do
  prefix="$directory/op-$operator"
  if [ "$(sha_of "$prefix.reference.f32")" != "$reference_sha" ] || [ "$(sha_of "$prefix.input.f32")" != "$input_sha" ]; then
    fail 'Probe fixture checksum mismatch.'
  fi
  log="$output/op-$operator.log"
  (
    export RUNE_LITERT_VALIDATION_DIAGNOSTICS=1 RUNE_LITERT_CANDIDATE_SHA256="$model_sha" RUNE_LITERT_PROBE_SHAPE="$shape"
    unset RUNE_LITERT_VALIDATE_NHWC RUNE_LITERT_VALIDATE_IR
    [ "$ir_graph" = true ] && export RUNE_LITERT_VALIDATE_IR=1
    "$checker" "$(native "$worker")" "$(native "$runtime")" "$(native "$prefix.tflite")" "$(native "$prefix")" </dev/null
  ) >"$log" 2>&1
  printf '%s\t%s\t%s\n' "$operator" "$?" "$(native "$log")" >>"$results"
done < <(printf '%s\n' "$manifest" | tail -n +2)

node -e '
  const fs = require("fs")
  const [manifestFile, resultsFile, out, workerSha, irGraph] = process.argv.slice(1)
  const proof = JSON.parse(fs.readFileSync(manifestFile, "utf8").replace(/^﻿/, ""))
  const rows = fs.readFileSync(resultsFile, "utf8").split("\n").filter(Boolean).map((line) => line.split("\t"))
  const probes = rows.map(([operator, exitCode, log]) => {
    const probe = proof.probes.find((p) => String(p.operator) === operator)
    const match = /max_absolute_error=([\d.Ee+-]+).*failed_values=(\d+)/.exec(fs.readFileSync(log, "utf8"))
    return {
      operator: probe.operator,
      kind: probe.kind,
      shape: probe.shape,
      passed: exitCode === "0",
      max_absolute_error: match ? Number(match[1]) : null,
      failed_values: match ? Number(match[2]) : null,
      exit_code: Number(exitCode),
      log,
    }
  })
  const report = {
    source_sha256: proof.source_sha256,
    worker_sha256: workerSha,
    graph: irGraph === "true" ? "ir" : "legacy",
    tolerance: { atol: 0.001, rtol: 0.001 },
    probes,
  }
  fs.writeFileSync(out, JSON.stringify(report, null, 2))
  for (const p of probes)
    console.log(`operator=${p.operator} kind=${p.kind} passed=${p.passed} max_error=${p.max_absolute_error ?? ""} failures=${p.failed_values ?? ""}`)
  console.log(`Report: ${out}`)
  if (probes.length !== proof.probes.length || probes.some((p) => !p.passed)) process.exit(1)
' "$(native "$directory/probes.json")" "$(native "$results")" "$(native "$output/report.json")" \
  "$(sha_of "$worker")" "$ir_graph" ||
  fail 'One or more diagnostic checkpoints failed; inspect the report.'
