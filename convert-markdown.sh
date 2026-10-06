#!/usr/bin/env bash

# Resolve the directory where the script is located (root directory)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

OUTPUT_FILE="Content.md"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# -----------------------------------------------------------------------------
# Extra patterns, files, or folders to ignore (in addition to .gitignore).
# Supports wildcards (*) and directory names.
# -----------------------------------------------------------------------------
EXTRA_IGNORES=(
    "$OUTPUT_FILE"
    "$SCRIPT_NAME"
    ".github"
    ".husky"
    ".vscode"
    "public"
    "capabilities"
    "bin"
    "gen"
    "icons"
    "bun.lock"
    "Cargo.lock"
    ".gitattributes"
    ".gitignore"
    "Dockerfile"
    "LICENSE"
    "nginx.conf"
    "index.html"
    "tsconfig.*"
    "env*"
    "commitlint.config.ts"
    "vite.config.ts"
    ".editorconfig"
    ".oxlintrc.json.bak"
    "prettier.config.js"
    "vitest.config.ts"
    "README.md"
    "AGENTS.md"
    "weights"
    "models.json"
    "smoke_test.json"
    "build.sh"
    "static"
    "commands"
    "registry"
    "scripts"
    "skills"
    "tasks"
    "lefthook.yml"
    "CHANGELOG"
    # "package.json"
    # "Cargo.toml"
)

# Helper function to check if a file matches any of the EXTRA_IGNORES patterns
is_extra_ignored() {
    local file="$1"
    local filename
    filename="$(basename "$file")"

    for pattern in "${EXTRA_IGNORES[@]}"; do
        # Matches exact path, filename glob, or subpath
        if [[ "$file" == $pattern ]] || \
           [[ "$filename" == $pattern ]] || \
           [[ "$file" == $pattern/* ]] || \
           [[ "$file" == */$pattern/* ]] || \
           [[ "$file" == */$pattern ]]; then
            return 0
        fi
    done
    return 1
}

# Clear/create the output file
> "$OUTPUT_FILE"

# Verify git repository exists
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: This script must be run inside a Git repository to parse .gitignore." >&2
    exit 1
fi

echo "Scanning repository from root: $ROOT_DIR"
echo "---------------------------------------------------------"

count=0

# Use process substitution (< <(...)) so the while loop runs in the current shell (enabling the counter)
while IFS= read -r -d '' file; do
    # Skip if file was deleted, is a directory, or matches extra ignore patterns
    if [[ ! -f "$file" ]] || is_extra_ignored "$file"; then
        continue
    fi

    # Only process text files (skips binaries, images, etc.)
    if file "$file" | grep -q 'text'; then
        # Log the file to the console
        echo " [+] $file"
        ((count++))

        {
            echo "---------------------------------------------------------"
            echo "// $file"
            cat "$file"
            echo "" # Ensures there's a newline before the closing separator
            echo "---------------------------------------------------------"
            echo ""
        } >> "$OUTPUT_FILE"
    fi
done < <(git ls-files -z --cached --others --exclude-standard | sort -z)

echo "---------------------------------------------------------"
echo "Done! Merged $count file(s) into $OUTPUT_FILE"