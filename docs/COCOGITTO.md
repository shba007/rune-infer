# Cocogitto release policy

Cocogitto is pinned to 7.0.0. It is the sole Conventional Commit classifier and
Markdown renderer for changelogs and release notes. The three Rune repositories
share commit types, bump flags, group order and `.release/changelog.tera`.
Only repository identity and Tools' independent package map differ.

Rust xtask selects reviewed history, excludes routine release/registry/format
bookkeeping and private planning messages, synchronizes manifests and validates
prepared metadata. Features advance minor; fixes/performance/semantic maintenance
advance patch. Breaking changes take precedence: minor below 1.0, major thereafter.
For pre-1.0 breaking fixes the integration requests Cocogitto's feature classification
for versioning only; rendered commits retain their original type and breaking marker.
Unknown valid semantic types use the configured maintenance bump policy.

The shared template uses Cocogitto's remote commit macro, emoji groups and dated
headings. It retains breaking body/footer descriptions and avoids inventing links
for historical notes without a known commit. Rendering is offline and dates derive
from reviewed Git timestamps. Release publication extracts the exact prepared
section; it never asks GitHub or another generator to create release prose.

The owner-authorized one-time format migration retains historical entry wording,
versions and available attribution. It updates local Markdown and receipts only;
it does not rewrite tags or already-published GitHub release bodies. Subsequent
preparation preserves historical sections. Agents leave work unstaged/uncommitted.

`cargo run --locked --manifest-path xtask/Cargo.toml --bin render-notes` is the
fixed-version formatting boundary. It reads JSON containing `base`, `version`,
`package`, `timestamp` and `commits` from stdin and writes Cocogitto Markdown.
Each commit contains `id`, `message` and `author` (`name`, `timestamp`).
This command does not write release files or mutate Git.
