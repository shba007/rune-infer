#!/usr/bin/env bash
set -uo pipefail

name="all"
build=false
validate_only=false
quiet=false
for argument in "$@"; do
  case "$argument" in
    --build) build=true ;;
    --validate-only) validate_only=true ;;
    --quiet) quiet=true ;;
    -*) echo "Unknown option '$argument'" >&2; exit 1 ;;
    *) name="$argument" ;;
  esac
done

cyan=$'\033[36m' gray=$'\033[90m' yellow=$'\033[33m' green=$'\033[32m' red=$'\033[31m' plain=$'\033[0m'

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
cd "$repo_root" || exit 1

echo "${cyan}=== Rune Infer Demo Recording Orchestrator ===${plain}"
echo "${gray}Repository root: $repo_root${plain}"

if ! command -v vhs >/dev/null 2>&1; then
  echo "Charm VHS CLI ('vhs') was not found on PATH." >&2
  echo 'Please install it: https://github.com/charmbracelet/vhs' >&2
  exit 1
fi

tapes=()
for tape in demos/*/*.tape; do
  [ -f "$tape" ] || continue
  case "$(basename "$(dirname "$tape")")" in common) continue ;; esac
  case "$(basename "$tape")" in *template*) continue ;; esac
  tapes+=("$tape")
done
if [ ${#tapes[@]} -eq 0 ]; then
  echo "warning: No active .tape files found in demos/*/" >&2
  exit 0
fi
demo_of() { basename "$(dirname "$1")"; }
names_of() { local list="" tape; for tape in "$@"; do list="${list:+$list, }$(demo_of "$tape")"; done; printf '%s' "$list"; }

if [ "$name" != "all" ]; then
  matched=()
  for tape in "${tapes[@]}"; do
    case "$(demo_of "$tape")" in *"$name"*) matched+=("$tape"); continue ;; esac
    case "$(basename "$tape" .tape)" in *"$name"*) matched+=("$tape") ;; esac
  done
  if [ ${#matched[@]} -eq 0 ]; then
    echo "No demo tape matching '$name'. Available demos: $(names_of "${tapes[@]}")" >&2
    exit 1
  fi
  tapes=("${matched[@]}")
fi
echo "${yellow}Selected demos: $(names_of "${tapes[@]}")${plain}"

if [ "$validate_only" = false ]; then
  bin_name="rune-infer"
  [ "${OS:-}" = "Windows_NT" ] && bin_name="rune-infer.exe"
  if [ "$build" = true ] || [ ! -f "target/release/$bin_name" ]; then
    echo
    echo "${cyan}Building release binary (target/release/$bin_name)...${plain}"
    cargo build --release --bin rune-infer || { code=$?; echo 'Release build failed.' >&2; exit "$code"; }
  fi
fi

clock() { printf '%02d:%02d' $(($1 / 60)) $(($1 % 60)); }

rows=()
for tape in "${tapes[@]}"; do
  demo="$(demo_of "$tape")"
  echo
  echo "${cyan}--> [$demo] Validating $tape...${plain}"
  vhs validate "$tape" || { code=$?; echo "Tape validation failed for $tape." >&2; exit "$code"; }
  if [ "$validate_only" = true ]; then
    echo "${green}    Validation passed.${plain}"
    continue
  fi

  echo "${cyan}--> [$demo] Rendering recording with VHS...${plain}"
  started=$SECONDS
  if [ "$quiet" = true ]; then vhs "$tape" -q; else vhs "$tape"; fi
  code=$?
  duration="$(clock $((SECONDS - started)))"

  if [ "$code" -eq 0 ]; then
    output_file="$(sed -nE 's/^[[:space:]]*Output[[:space:]]+(.+)$/\1/p' "$tape" | head -n 1 | tr -d '\r' | sed -E 's/[[:space:]]+$//')"
    size=""
    if [ -n "$output_file" ] && [ -f "$output_file" ]; then
      size="$(awk -v bytes="$(wc -c <"$output_file")" 'BEGIN { printf "%.2f MB", bytes / 1048576 }')"
    else
      output_file=""
    fi
    echo "${green}    Completed in $duration -> $output_file ($size)${plain}"
    rows+=("$demo|Success|$duration|$output_file|$size")
  else
    echo "${red}    Recording failed with exit code $code${plain}"
    rows+=("$demo|Failed|$duration|N/A|N/A")
  fi
done

if [ "$validate_only" = false ] && [ ${#rows[@]} -gt 0 ]; then
  echo
  echo "${cyan}=== Recording Summary ===${plain}"
  printf '%-12s %-8s %-9s %-28s %s\n' Demo Status Duration Output Size
  for row in "${rows[@]}"; do
    IFS='|' read -r demo status duration output_file size <<<"$row"
    printf '%-12s %-8s %-9s %-28s %s\n' "$demo" "$status" "$duration" "$output_file" "$size"
  done
fi
