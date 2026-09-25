#!/usr/bin/env bash

# Use the provided directory or default to the current directory
TARGET_DIR="${1:-./src}"
OUTPUT_FILE="Content.md"

# Clear/create the output file
> "$OUTPUT_FILE"

# Find all regular files recursively:
# - Excludes the output file itself
# - Excludes hidden directories/files (like .git, .node_modules, etc.)
find "$TARGET_DIR" -type f ! -name "$OUTPUT_FILE" ! -path '*/.*' | sort | while IFS= read -r file; do
    # Only process text files (skips binaries, images, etc.)
    if file "$file" | grep -q 'text'; then
        {
            echo "---------------------------------------------------------"
            echo "// $file"
            cat "$file"
            echo "" # Ensures there's a newline before the closing separator
            echo "---------------------------------------------------------"
            echo ""
        } >> "$OUTPUT_FILE"
    fi
done

echo "Done! Text merged into $OUTPUT_FILE"