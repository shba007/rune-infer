#!/bin/sh
# Rune Infer installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/shba007/rune-infer/main/install.sh | sh
#
# Environment overrides:
#   RUNE_VERSION      release tag to install (default: latest), e.g. v0.2.0
#   RUNE_INSTALL_DIR  where to put the binary (default: ~/.local/bin)
#   RUNE_REPO         GitHub repo (default: shba007/rune-infer)
#   RUNE_BASE_URL     download base URL or local directory (mirrors / offline installs)
#   RUNE_PLATFORM     force a platform name, e.g. linux-x64 (advanced)
#   RUNE_SKIP_VERIFY  set to 1 to skip the SHA-256 check (not recommended)
set -eu

REPO="${RUNE_REPO:-shba007/rune-infer}"
VERSION="${RUNE_VERSION:-latest}"
INSTALL_DIR="${RUNE_INSTALL_DIR:-$HOME/.local/bin}"
BIN="rune-infer"

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- platform ---------------------------------------------------------------
detect_platform() {
  if [ -n "${RUNE_PLATFORM:-}" ]; then printf '%s' "$RUNE_PLATFORM"; return; fi
  os=$(uname -s)
  arch=$(uname -m)
  case "$os" in
    Linux)  os=linux ;;
    Darwin) os=macos ;;
    *) die "unsupported OS '$os'. On Windows use install.ps1; otherwise build from source: cargo install --git https://github.com/$REPO" ;;
  esac
  case "$arch" in
    x86_64|amd64)  arch=x64 ;;
    arm64|aarch64) arch=arm64 ;;
    *) die "unsupported CPU architecture '$arch'" ;;
  esac
  printf '%s-%s' "$os" "$arch"
}

PLATFORM=$(detect_platform)
case "$PLATFORM" in
  linux-x64|macos-x64|macos-arm64) ;;
  *) die "no prebuilt binary for '$PLATFORM'. Build from source: cargo install --git https://github.com/$REPO" ;;
esac

ARCHIVE="$BIN-$PLATFORM.tar.gz"

if [ -n "${RUNE_BASE_URL:-}" ]; then
  BASE="${RUNE_BASE_URL%/}"
elif [ "$VERSION" = "latest" ]; then
  BASE="https://github.com/$REPO/releases/latest/download"
else
  case "$VERSION" in v*) ;; *) VERSION="v$VERSION" ;; esac
  BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

# ---- download helpers -------------------------------------------------------
fetch() { # fetch <name> <dest>
  src="$BASE/$1"
  if [ -d "$BASE" ]; then
    [ -f "$src" ] || die "$src not found"
    cp "$src" "$2"
  elif have curl; then
    curl -fsSL --retry 3 -o "$2" "$src" || die "download failed: $src"
  elif have wget; then
    wget -q -O "$2" "$src" || die "download failed: $src"
  else
    die "need curl or wget to download files"
  fi
}

sha256_of() {
  if have sha256sum; then sha256sum "$1" | cut -d' ' -f1
  elif have shasum; then shasum -a 256 "$1" | cut -d' ' -f1
  elif have openssl; then openssl dgst -sha256 "$1" | sed 's/^.*= //'
  else return 1
  fi
}

have tar || die "need tar to unpack the download"

TMP=$(mktemp -d 2>/dev/null || mktemp -d -t rune-infer)
trap 'rm -rf "$TMP"' EXIT INT TERM

say "Installing Rune Infer ($PLATFORM, $VERSION)..."
fetch "$ARCHIVE" "$TMP/$ARCHIVE"

# ---- integrity --------------------------------------------------------------
if [ "${RUNE_SKIP_VERIFY:-0}" = "1" ]; then
  warn "skipping checksum verification"
else
  fetch "$ARCHIVE.sha256" "$TMP/$ARCHIVE.sha256"
  expected=$(cut -d' ' -f1 "$TMP/$ARCHIVE.sha256" | tr 'A-F' 'a-f')
  actual=$(sha256_of "$TMP/$ARCHIVE") || die "no sha256 tool found (install coreutils/openssl, or set RUNE_SKIP_VERIFY=1)"
  [ "$expected" = "$actual" ] || die "checksum mismatch for $ARCHIVE (expected $expected, got $actual) — aborting"
  say "Checksum verified."
fi

# ---- install ----------------------------------------------------------------
tar -xzf "$TMP/$ARCHIVE" -C "$TMP" || die "could not unpack $ARCHIVE"
[ -f "$TMP/$BIN" ] || die "archive did not contain '$BIN'"

mkdir -p "$INSTALL_DIR" || die "cannot create $INSTALL_DIR (set RUNE_INSTALL_DIR to a writable folder)"
upgrade=0; [ -e "$INSTALL_DIR/$BIN" ] && upgrade=1
# Write beside the target then rename, so a running instance is not corrupted.
cp "$TMP/$BIN" "$INSTALL_DIR/$BIN.new" && chmod 755 "$INSTALL_DIR/$BIN.new" \
  && mv -f "$INSTALL_DIR/$BIN.new" "$INSTALL_DIR/$BIN" \
  || die "cannot write to $INSTALL_DIR"

if [ "$PLATFORM" != "${PLATFORM#macos}" ] && have xattr; then
  xattr -d com.apple.quarantine "$INSTALL_DIR/$BIN" 2>/dev/null || true
fi

if [ "$upgrade" = 1 ]; then say "Upgraded: $INSTALL_DIR/$BIN"; else say "Installed: $INSTALL_DIR/$BIN"; fi
"$INSTALL_DIR/$BIN" --version 2>/dev/null || warn "installed binary did not start; run '$INSTALL_DIR/$BIN --version' to see why"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    say ""
    say "$INSTALL_DIR is not on your PATH. Add it with:"
    case "${SHELL:-}" in
      */zsh)  say "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc && exec zsh" ;;
      */fish) say "  fish_add_path $INSTALL_DIR" ;;
      *)      say "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.profile && . ~/.profile" ;;
    esac
    ;;
esac

say ""
say "Next: run 'rune-infer' — a short setup wizard creates your config in ~/.rune-infer."
say "Optional: install FFmpeg for video input."
