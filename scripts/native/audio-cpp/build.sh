#!/usr/bin/env bash
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_tree=""
build_directory="target/audiocpp-patched"
backend="cuda"
cuda_architectures="native"
prebuilt=""
allow_unsupported=false
jobs=6
while [ $# -gt 0 ]; do
  case "$1" in
    --source-tree) source_tree="$2"; shift 2 ;;
    --build-directory) build_directory="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --cuda-architectures) cuda_architectures="$2"; shift 2 ;;
    --prebuilt-backend-directory) prebuilt="$2"; shift 2 ;;
    --allow-unsupported-cuda-compiler) allow_unsupported=true; shift ;;
    --jobs) jobs="$2"; shift 2 ;;
    *) echo "Unknown option '$1'" >&2; exit 1 ;;
  esac
done
fail() { echo "$1" >&2; exit 1; }
[ -n "$source_tree" ] || fail '--source-tree is required'
case "$backend" in cpu | cuda) ;; *) fail '--backend must be cpu or cuda' ;; esac

revision='41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48'
actual="$(git -C "$source_tree" rev-parse HEAD 2>/dev/null)" || actual=""
[ "$actual" = "$revision" ] || fail "Expected audio.cpp $revision"

patch="$here/gguf-magic.patch"
command -v cygpath >/dev/null 2>&1 && patch="$(cygpath -m "$patch")"
if ! git -C "$source_tree" apply --reverse --check "$patch" 2>/dev/null; then
  git -C "$source_tree" apply --check "$patch" || fail 'Source conflicts with the GGUF patch; no changes applied'
  git -C "$source_tree" apply "$patch" || fail 'Could not apply the GGUF patch'
fi

cuda=OFF
[ "$backend" = cuda ] && cuda=ON
cuda_flags=()
[ "$allow_unsupported" = true ] && cuda_flags+=('-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler')
reuse=OFF
if [ -n "$prebuilt" ]; then
  cli="$prebuilt/audiocpp_cli"
  [ -f "$cli.exe" ] && cli="$cli.exe"
  version="$("$cli" --version 2>&1)" || fail 'Prebuilt backends must come from the pinned Windows audio.cpp release'
  short="$(printf '%s\n' "$version" | sed -nE 's/.*git:[[:space:]]+([a-f0-9]{7,40}).*/\1/p' | head -n 1)"
  case "$revision" in
    "$short"*) [ -n "$short" ] || fail 'Prebuilt backends must come from the pinned Windows audio.cpp release' ;;
    *) fail 'Prebuilt backends must come from the pinned Windows audio.cpp release' ;;
  esac
  reuse=ON
fi

cmake -S "$source_tree" -B "$build_directory" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  '-DAUDIOCPP_VERSION:STRING=0.9.3 (rune-gguf1)' -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
  "-DENGINE_ENABLE_CUDA=$cuda" -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON \
  -DENGINE_BUILD_TESTS=ON "-DCMAKE_CUDA_ARCHITECTURES=$cuda_architectures" \
  "-DAUDIOCPP_PREBUILT_BACKENDS=$reuse" ${cuda_flags[@]+"${cuda_flags[@]}"} ||
  fail 'audio.cpp configuration failed'
cmake --build "$build_directory" --target audiocpp_cli gguf_tensor_source_test --parallel "$jobs" ||
  fail 'audio.cpp build failed'
if [ -n "$prebuilt" ]; then
  for library in "$prebuilt"/*.dll; do
    [ -f "$library" ] && cp "$library" "$build_directory/bin/"
  done
fi
ctest --test-dir "$build_directory" -R '^gguf_tensor_source_test$' --output-on-failure ||
  fail 'GGUF format regression failed'
