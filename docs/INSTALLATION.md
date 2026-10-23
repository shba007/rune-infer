# Installation, user data and release preparation

Default layout on Windows, Linux and macOS:

```text
<user-home>/rune/
  rune-infer[.exe]
  rune-kit[.exe]              # owned by Kit, not modified by Infer
  tools/plugins/             # owned by Kit, not modified by Infer
  infer/
    config/server.json       # schema 3 server settings
    config/catalog.json      # schema 3 model catalog
    .env                     # user secrets; not printed by home migration
    bin/                     # managed inference engines
    logs/                    # logs and fresh process state
```

Installers verify archive checksums and validate the candidate executable before
replacing Infer. Explicit `RUNE_INSTALL_DIR`, `RUNE_VERSION`, `RUNE_REPO`,
`RUNE_BASE_URL`, platform and checksum overrides remain supported. The existing
Infer executable must identify itself before an update; an unrecognized
destination is a conflict. Old executables in `~/.local/bin` or
`%LOCALAPPDATA%/Programs/rune-infer` are retained. Remove old PATH entries only
after checking `rune-infer --version` resolves the intended executable.

The shared install directory is added at most once to user PATH. PowerShell
updates user PATH; the POSIX installer adds a guarded entry to `.profile` or
`.zprofile` for zsh. Start a new login shell, or source the applicable profile.
`RUNE_NO_MODIFY_PATH=1` disables PATH changes on either platform. Other shell
startup conventions may require adding `~/rune` manually.

## Home selection

Precedence is unchanged: explicit `--config` > `RUNE_HOME` > a local development
`./config/server.json` or `./config/models.json` > `~/rune/infer`.
`--config` accepts an existing home/config directory or a configuration file.
`RUNE_HOME` is the Infer home itself, **not** the shared parent. Relative explicit
paths resolve against the initial working directory. Missing user-home resolution
is an error, with instructions to select an explicit path; it never selects `.`.

Development checkout behavior is intentional. To use persistent configuration
while running from this repository, set `RUNE_HOME` explicitly. `.env` is loaded
only from the selected home, so an unrelated parent directory's `.env` is not
silently used. The setup wizard shows this home and writes its `config/` files.

## Legacy home migration

On implicit default startup, if `~/.rune-infer/config` contains configuration:

1. Copy `config/` (including custom files), `.env`, `bin/` and `logs/` into a
   uniquely owned staging directory beside `~/rune/infer`. Links/junctions are
   rejected, not followed. The old `.active_pids` registry is not copied.
2. Convert schema 1/2 `config/models.json` to schema 3 **in the copy only**.
   Environment templates remain templates, and the copied `.env` stays unchanged.
3. Rebase relative `path` / `*_path` values to the original home. Existing weights
   and external/custom referenced files stay where they are; absolute references
   stay absolute. Opaque engine argument strings are preserved verbatim, so any
   relative filenames embedded in those arguments require manual review.
4. Validate the copied configuration using its environment file and process
   environment. Validation failures do not print config/secret values and do not
   select the staged home. Check schema/environment values in the original.
5. Confirm source config/environment did not change, write a migration receipt,
   then rename the validated staged directory to `~/rune/infer`.

Original files and directories are never deleted or modified. Unrelated files,
projects and model directories are not copied. A receipt makes a repeat launch
safe and preserves edits made in the destination. If legacy config/environment
changes after migration, startup reports a conflict instead of overwriting either
copy. A pre-existing destination without the matching receipt is also a conflict.

To resolve a conflict, compare the two homes manually and choose one with
`RUNE_HOME` or `--config`; explicit selection bypasses automatic home migration.
Do not delete the old home while configuration still references its weights or
custom files. Standalone `rune-infer migrate` remains the explicit schema-upgrade
command; it is separate from automatic home migration.

## Release preparation

The owner reviews and commits changes on develop, promotes to main, then creates
and pushes the matching version tag. Agents must leave changes uncommitted and
must not stage, commit, push, merge, create tags, publish or dispatch workflows.

Prepare Cargo.toml/Cargo.lock and the matching CHANGELOG section before committing.
New public descriptions cover functionality, compatibility and migration, without
internal planning labels. Preview without publishing:

```text
python scripts/release_notes.py --version v0.8.0 --output target/release-notes-preview.md
```

The script extracts that exact version section; missing, empty, mismatched,
duplicate or placeholder sections fail. CI uploads this preview and runs checks
without writing source commits or pushing branches. The former formatting
auto-commit and version-update PR/push steps have been removed.

After the **owner** pushes an existing version tag on main, release automation
checks main ancestry and version/changelog consistency and uses `gh release create
--verify-tag` with the preview file. It does not create tags. Repository `GH_TOKEN`
must permit release publication and trigger the artifact workflow. Artifact upload
does not overwrite existing assets. Historical releases, tags and changelog
sections are not regenerated.

The proposed 0.8.0 also includes the already committed Omni discovery/projector
changes after 0.7.0. Installation regression tests do not validate live Omni audio,
vision, speech output, quality or co-residency; those remain separate acceptance
work. Do not infer their readiness from a successful text-model migration trial.

## Protocol boundary

This repository has no public MCP server/client, MCP transport, JSON-RPC lifecycle,
or plugin ABI. Its `/v1` inference HTTP routes are OpenAI-compatible/Rune APIs;
tool-call JSON in a completion is not an MCP tools transport. The ecosystem's
[MCP 2026-07-28 target](https://modelcontextprotocol.io/specification/2026-07-28)
therefore does not change Infer's HTTP API version or add protocol capabilities.
Any future real MCP surface must implement that revision's schema and transport
requirements separately.
