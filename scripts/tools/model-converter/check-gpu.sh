#!/usr/bin/env bash
# Windows GPU (WebGPU on Direct3D12) parity check of a depth model against a fixture.
#
#   bash scripts/tools/model-converter/check-gpu.sh --model <file> --runtime-directory <dir>
#        --shader-directory <dir> --fixture-prefix <prefix> [--conversion-report <report.json>]
#        [--ir-graph]
#
# Run from Git Bash on Windows. Needs cargo, rustc and node (node only reads and writes JSON).
# Deliberately no downloads, installation, catalog edits or Python dependency.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$here/../../.." && pwd)"
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
absolute() { printf '%s/%s' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
fail() { echo "$1" >&2; exit 1; }

model="" runtime_directory="" shader_directory="" fixture_prefix="" conversion_report="" ir_graph=false
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --runtime-directory) runtime_directory="$2"; shift 2 ;;
    --shader-directory) shader_directory="$2"; shift 2 ;;
    --fixture-prefix) fixture_prefix="$2"; shift 2 ;;
    --conversion-report) conversion_report="$2"; shift 2 ;;
    --ir-graph) ir_graph=true; shift ;;
    *) fail "Unknown option '$1'" ;;
  esac
done
for required in model runtime_directory shader_directory fixture_prefix; do
  [ -n "${!required}" ] || fail "--${required//_/-} is required"
done
[ -f "$model" ] || fail "No model at $model"
[ -f "$runtime_directory/libLiteRt.dll" ] || fail "No libLiteRt.dll in $runtime_directory"
[ -d "$shader_directory" ] || fail "No shader directory at $shader_directory"
[ -f "$fixture_prefix.input.f32" ] || fail "No $fixture_prefix.input.f32"
[ -f "$fixture_prefix.reference.f32" ] || fail "No $fixture_prefix.reference.f32"
model="$(absolute "$model")"
runtime="$(absolute "$runtime_directory/libLiteRt.dll")"
shaders="$(cd "$shader_directory" && pwd)"
input="$(absolute "$fixture_prefix.input.f32")"
reference="$(absolute "$fixture_prefix.reference.f32")"
prefix="${input%.input.f32}"

expected_hash='6f03564f13abfae18185581a809cb5c0ee5c0a425dc5f85102194b2b321eba3d'
nhwc=false
if [ -n "$conversion_report" ]; then
  [ -f "$conversion_report" ] || fail "No conversion report at $conversion_report"
  conversion_report="$(absolute "$conversion_report")"
  # Prints: <candidate sha256> <true|false for NHWC input>
  proof="$(node -e '
    const proof = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8").replace(/^﻿/, ""))
    const checks = Array.isArray(proof.checks) ? proof.checks : []
    const sha = String(proof.candidate?.sha256 ?? "")
    if (proof.status !== "linux_cpu_parity_passed" || checks.length < 1 || checks.some((c) => !c.passed) || !/^[a-f0-9]{64}$/.test(sha)) {
      console.error("A completed CPU parity report with a valid candidate SHA-256 is required.")
      process.exit(1)
    }
    console.log(sha + "\t" + (proof.input?.layout === "NHWC"))
  ' "$(native "$conversion_report")")" || exit 1
  IFS=$'\t' read -r expected_hash nhwc <<<"$proof"
fi

cd "$repository" || exit 1
mkdir -p target
# GPU worker verifies pinned runtime, WebGPU and DXC DLLs before loading them.
directory="$(mktemp -d "$repository/target/litert-gpu-check-XXXXXXXXXXXX")"
for name in dxcompiler.dll dxil.dll; do
  cp "$shaders/$name" "$directory/" || fail "No $name in $shaders"
done

build="$(CARGO_HUSKY_DONT_INSTALL_HOOKS=1 cargo build --locked --lib --target-dir target/ci-check --message-format=json)" ||
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
  // Cargo may emit mixed Windows separators; use one form so rustc resolves the
  // extern crate and its transitive metadata consistently.
  console.log(require("path").resolve(anyhow[0]) + "\t" + require("path").resolve(sha2[0]))
')" || exit 1
IFS=$'\t' read -r anyhow sha2 <<<"$rlibs"

worker="$directory/gpu-worker.exe"
checker="$directory/native-check.exe"
rustc --edition 2024 -D warnings "$(native "$here/gpu_worker.rs")" \
  -L "dependency=$(native "$repository/target/ci-check/debug/deps")" --extern "sha2=$sha2" \
  --extern "anyhow=$anyhow" -o "$(native "$worker")" || fail 'GPU validation worker build failed.'
rustc --edition 2024 -D warnings "$(native "$here/native_check.rs")" -o "$(native "$checker")" ||
  fail 'Native checker build failed.'

log="$directory/validation.log"
(
  export RUNE_LITERT_CANDIDATE_SHA256="$expected_hash" RUNE_LITERT_VALIDATION_DIAGNOSTICS=1
  unset RUNE_LITERT_VALIDATE_IR RUNE_LITERT_VALIDATE_NHWC
  [ "$ir_graph" = true ] && export RUNE_LITERT_VALIDATE_IR=1
  [ "$nhwc" = true ] && export RUNE_LITERT_VALIDATE_NHWC=1
  "$checker" "$(native "$worker")" "$(native "$runtime")" "$(native "$model")" "$(native "$prefix")"
) >"$log" 2>&1
exit_code=$?

node -e '
  const [out, exitCode, irGraph, nhwc, report, model, modelSha, input, inputSha, reference, referenceSha, log] = process.argv.slice(1)
  const result = {
    status: exitCode === "0" ? "gpu_fp32_parity_passed" : "gpu_validation_failed",
    backend: "Windows WebGPU Direct3D12, GPU with CPU operations permitted",
    graph: irGraph === "true" ? "ir" : "legacy",
    conversion_report: report || null,
    nhwc_boundaries: nhwc === "true",
    tolerance: { atol: 0.001, rtol: 0.001 },
    model: { path: model, sha256: modelSha },
    input: { path: input, sha256: inputSha },
    reference: { path: reference, sha256: referenceSha },
    log,
    exit_code: Number(exitCode),
  }
  require("fs").writeFileSync(out, JSON.stringify(result, null, 2))
' "$(native "$directory/report.json")" "$exit_code" "$ir_graph" "$nhwc" \
  "$([ -n "$conversion_report" ] && native "$conversion_report")" \
  "$(native "$model")" "$(sha_of "$model")" "$(native "$input")" "$(sha_of "$input")" \
  "$(native "$reference")" "$(sha_of "$reference")" "$(native "$log")" || fail 'Could not write the validation report.'

echo "Validation report: $directory/report.json"
[ "$exit_code" -eq 0 ] || fail "GPU validation failed; inspect $log. Production configuration was not changed."
