#!/bin/sh
# Rune Infer installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/shba007/rune-infer/main/install.sh | sh
#
# Environment overrides:
#   RUNE_VERSION      release tag to install (default: latest), e.g. v0.2.0
#   RUNE_INSTALL_DIR  where to put the binary (default: ~/rune)
#   RUNE_REPO         GitHub repo (default: shba007/rune-infer)
#   RUNE_BASE_URL     download base URL or local directory (mirrors / offline installs)
#   RUNE_PLATFORM     force a platform name, e.g. linux-x64 (advanced)
#   RUNE_SKIP_VERIFY  set to 1 to skip the SHA-256 check (not recommended)
set -eu

REPO="${RUNE_REPO:-shba007/rune-infer}"
VERSION="${RUNE_VERSION:-latest}"
if [ -z "${RUNE_INSTALL_DIR:-}" ]; then
  case "${HOME:-}" in /*) ;; *) printf '%s\n' 'error: cannot resolve user home; set RUNE_INSTALL_DIR' >&2; exit 1 ;; esac
fi
INSTALL_DIR="${RUNE_INSTALL_DIR:-${HOME}/rune}"
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
  linux-x64|linux-arm64|macos-x64|macos-arm64) ;;
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
KEEP_TMP=0
trap 'if [ "$KEEP_TMP" = 0 ]; then rm -rf "$TMP"; else warn "backup retained in $TMP"; fi' EXIT INT TERM

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

chmod 755 "$TMP/$BIN"
candidate=$("$TMP/$BIN" --version) || die "downloaded executable did not start; installation unchanged"
case "$candidate" in 'rune-infer '[0-9]*.[0-9]*.[0-9]*) ;; *) die "downloaded executable has an unexpected identity" ;; esac
[ "$VERSION" = latest ] || [ "$candidate" = "rune-infer ${VERSION#v}" ] || die "downloaded executable does not match requested version"
mkdir -p "$INSTALL_DIR" || die "cannot create $INSTALL_DIR (set RUNE_INSTALL_DIR)"
upgrade=0
[ ! -L "$INSTALL_DIR/$BIN" ] || die "destination is a link; it was not overwritten"
if [ -e "$INSTALL_DIR/$BIN" ]; then
  upgrade=1
  prior=$("$INSTALL_DIR/$BIN" --version) || die "destination conflict: existing executable did not validate"
  case "$prior" in 'rune-infer '[0-9]*.[0-9]*.[0-9]*) ;; *) die "destination conflict: existing file was not overwritten" ;; esac
  cp -p "$INSTALL_DIR/$BIN" "$TMP/previous" || die "cannot preserve existing executable"
fi
# mktemp prevents overwriting an unrelated .new file or the other product.
STAGED=$(mktemp "$INSTALL_DIR/.rune-infer-install.XXXXXX") || die "cannot stage executable"
if ! { cp "$TMP/$BIN" "$STAGED" && chmod 755 "$STAGED" && mv -f "$STAGED" "$INSTALL_DIR/$BIN"; }; then
  rm -f "$STAGED"
  die "cannot install executable"
fi
installed=$("$INSTALL_DIR/$BIN" --version) || installed=''
if [ "$installed" != "$candidate" ]; then
  if [ "$upgrade" = 1 ]; then
    if ! cp -p "$TMP/previous" "$INSTALL_DIR/$BIN"; then KEEP_TMP=1; die "cannot restore old executable; backup retained"; fi
  else rm -f "$INSTALL_DIR/$BIN"; fi
  die "installed executable failed validation; previous installation restored"
fi

if [ "$PLATFORM" != "${PLATFORM#macos}" ] && have xattr; then
  xattr -d com.apple.quarantine "$INSTALL_DIR/$BIN" 2>/dev/null || true
fi

if [ "$upgrade" = 1 ]; then say "Upgraded: $INSTALL_DIR/$BIN"; else say "Installed: $INSTALL_DIR/$BIN"; fi
say "$installed"

# Persistent PATH entry with a runtime guard; repeated installs and Kit coexistence
# do not append another copy. Explicit opt-out leaves user profiles untouched.
if [ "${RUNE_NO_MODIFY_PATH:-0}" != 1 ]; then
  case "${HOME:-}" in /*) ;; *) die "cannot resolve home for PATH update; set RUNE_NO_MODIFY_PATH=1" ;; esac
  case "${SHELL:-}" in */zsh) PROFILE="$HOME/.zprofile" ;; *) PROFILE="$HOME/.profile" ;; esac
  quoted=$(printf '%s' "$INSTALL_DIR" | sed "s/'/'\\\\''/g")
  if ! grep -Fq "$INSTALL_DIR" "$PROFILE" 2>/dev/null && ! grep -Fq "_rune_path='$quoted'" "$PROFILE" 2>/dev/null && ! { [ "$INSTALL_DIR" = "$HOME/rune" ] && grep -Fq '$HOME/rune' "$PROFILE" 2>/dev/null; }; then
    printf '\n# Rune shared PATH\n_rune_path='\''%s'\''\ncase ":$PATH:" in *":$_rune_path:"*) ;; *) export PATH="$_rune_path:$PATH" ;; esac\nunset _rune_path\n' "$quoted" >> "$PROFILE"
  fi
fi
say ""
say "Next: run 'rune-infer setup'; configuration lives in ${RUNE_HOME:-${HOME:-<user-home>}/rune/infer}/config."
say "Legacy configuration is copied and validated on first launch; originals and old executable locations are retained."
say "Optional: install FFmpeg for video input."
