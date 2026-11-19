#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$here/../../.." && pwd)"
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

images=()
arguments=(--output-dir /work/output)
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-linear-conv | --gpu-nhwc-boundaries | --gpu-fold-layer-scales) arguments+=("$1") ;;
    -*) echo "Unknown option '$1'" >&2; exit 1 ;;
    *) images+=("$1") ;;
  esac
  shift
done
[ ${#images[@]} -gt 0 ] || { echo 'Give at least one image file.' >&2; exit 1; }
for image in "${images[@]}"; do
  [ -f "$image" ] || { echo "Not an image file: $image" >&2; exit 1; }
done

mkdir -p "$repository/target"
work="$(mktemp -d "$repository/target/depth-large-XXXXXXXXXXXX")"
index=0
for image in "${images[@]}"; do
  file="$(basename "$image")"
  extension=""
  case "$file" in *.*) extension=".${file##*.}" ;; esac
  name="image-$index$extension"
  cp "$image" "$work/$name"
  arguments+=(--image "/work/$name")
  index=$((index + 1))
done

export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

docker build -t rune-infer-depth-converter:local "$(native "$here")" ||
  { echo 'Converter image build failed.' >&2; exit 1; }
docker run --rm --memory=12g --memory-swap=16g --cpus=4 --cap-drop=ALL \
  --security-opt=no-new-privileges --mount "type=bind,source=$(native "$work"),target=/work" \
  rune-infer-depth-converter:local "${arguments[@]}" ||
  { echo "Conversion failed. Retained artifacts: $work" >&2; exit 1; }
echo "Candidate and validation report: $work/output"
echo 'Windows native validation is still required before registering this artifact.'
