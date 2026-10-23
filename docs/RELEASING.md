# Release preparation and publication

Agents implement/test locally and leave changes unstaged/uncommitted. The owner
commits/pushes development work and reviews/merges release PRs. GitHub Actions is
authorized to calculate versions, update release files, commit/push the dedicated
release branch, open/update its PR, then create tags/releases/artifacts after owner
promotion. No AI service is involved.

## Event sequence and owner steps

1. Owner pushes reviewed conventional commits to `develop`. Rust CI validates the
   application and release utility on Linux and Windows: build, tests, Clippy,
   formatting, installer/migration fixtures and matching current release notes.
   PRs targeting develop/main and pushes to release/update-version also validate.
   There are no blanket exclusions for merge/release commits.
2. Only a successful trusted develop push can run preparation, after **both**
   validation jobs. The serialized job refuses a stale tested source SHA. Rust
   `prepare --apply` updates Cargo.toml, Cargo.lock, CHANGELOG.md and
   .release/prepared.json; it invokes no Git mutations. Generated files receive
   locked metadata/build/tests/Clippy/format checks before PR creation. Source
   validation receives no release secrets.
3. Actions commits/pushes only those four release files to the fixed branch
   `release/update-version`, targeting develop. Commit and PR title are exactly
   `ci(release): update version`. Re-runs update the same outstanding PR. No
   releasable changes means no PR. The branch push receives ordinary Linux/Windows
   validation, so generated changes are checked at their actual commit SHA too.
4. Owner reviews the notes, versions and checks, then merges the preparation PR
   into develop. **Merge or squash is supported** for this preparation PR; its
   consumed source SHA remains an ancestor either way. Do not squash unrelated
   product changes into the bookkeeping subject before they are prepared.
5. Owner opens a develop → main promotion PR. Use **Create a merge commit**,
   preserving the reviewed ancestry; do not squash or rebase promotion. Set its
   main merge commit title to `ci(release): publish vX.Y.Z`, using the prepared
   version. Merging into develop alone does not publish. Main publication checks
   the exact marker, two-parent merge, prepared version and source ancestry.
6. Rust Release on that main push extracts the exact prepared version section,
   refuses existing tags, then creates the matching tag/release at the main SHA.
   It uses GH_TOKEN so `release.published` starts Rust CD. CD checks out the tag,
   validates the event tag/body against the same extractor, then uploads platform
   archives/checksums and bootstrap installers. Existing assets are not overwritten.
7. Owner checks the actual Actions runs, published notes, Windows archive/version,
   checksums and installation before treating delivery as complete. Platform
   builds are independent; Windows ARM remains an existing experimental build.

## Deterministic rules and repeat preparation

The small standalone Rust xtask uses git-conventional, semver and toml_edit.
Infer retains its previous cliff bump rules: feature → minor, breaking → major
(also before 1.0), otherwise patch. All valid semantic commit types can prepare
changes, including docs/refactor/style/test/build/ci/chore/revert/custom. Reviewed
subjects become notes under Features/Fixes/Performance/Maintenance/Compatibility.
No prose is invented. Routine release bumps/autoformat bookkeeping, invalid
messages, placeholders and internal planning labels are excluded.

Fetched version tags establish the published baseline. Publication tags main's
promotion merge, which can be ahead of develop; the tag-to-develop range still
excludes the reviewed develop history preserved through that merge. An already pending
version is reused or advanced if new changes require it. Existing pending notes
are retained; identical new entries are deduplicated. Published sections remain
byte-for-byte intact. The first preparation can retain the current unpublished
0.8.0 installation/migration guidance and Omni limitations while adding reviewed
commits since 0.7.0. This does not establish live Omni audio/vision quality.

.release/prepared.json records the consumed source head and version. After the
preparation PR merges, its bookkeeping commit produces no further PR; new reviewed
commits extend the same pending release. A history rewrite or mismatched state
fails clearly. Repeating preparation with the same inputs gives the same version
and changelog, without new sections or repeated bumps.

Publication **never generates new history prose**. `notes` extracts the complete
matching section (heading included), retaining its content with canonical LF line
endings and one final newline on every checkout platform.
Missing, mismatched, duplicate, empty, placeholder or private notes fail. CD rejects
an event whose version/body differs from the prepared section. Historical tags,
commits and releases are not rewritten.

## Local previews

From the repository root:

```text
cargo test --locked --manifest-path xtask/Cargo.toml
cargo clippy --locked --manifest-path xtask/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path xtask/Cargo.toml --check
cargo run --locked --manifest-path xtask/Cargo.toml -- prepare
cargo run --locked --manifest-path xtask/Cargo.toml -- example
cargo run --locked --manifest-path xtask/Cargo.toml -- notes
```

Build the main binary first for native Windows installer acceptance:
`cargo build --locked`. Default prepare/example/notes never mutate Git or release
files. `notes --out target/release-notes-preview.md` optionally writes a preview.
`--apply` is reserved for authorized CI/owner preparation; agents must not run it
on the working repository. Tests exercise apply only inside isolated fixtures.

The synthetic example is deliberately independent of local history:

```markdown
## [0.8.1]

### Fixes

- install: preserve existing configuration during updates
```

The actual local dry run currently reuses pending **0.8.0** and includes reviewed
Omni detection/projector changes and the migration trial helper. It will also
include the owner's future conventional commit for this tooling change. No version
or generated changelog was applied by the agent.

## Credentials and repository settings

Preparation uses GITHUB_TOKEN for PR API operations (job Pull requests write and
Contents write), and the existing GH_TOKEN secret for branch pushes. GH_TOKEN
must be a PAT or GitHub App credential with Contents write for this repository and
events that trigger workflows; it need not have Pull requests write for this
split. Release creation also uses GH_TOKEN for downstream `release.published`.
Artifact uploads use GITHUB_TOKEN. Verify token expiry, repository/organization
approval/SSO and any workflow-file permission required by the push ancestry. A
short-lived App credential must be refreshed through the existing integration;
an expired static installation token will not work. Missing credentials fail
explicitly; no secret values are logged.

[GitHub's event rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
restrict workflows from GITHUB_TOKEN events, and bot PR events can require
approval. The [create-pull-request credential split](https://github.com/peter-evans/create-pull-request)
lets the GH_TOKEN branch push start validation independently of bot PR approval.
Generated-file checks also run before opening the PR. Owner approval may still be
needed for its PR-event run; never assume bot PR creation validated the branch.

Repository/org policy must allow Actions and the listed actions, allow Actions to
create PRs, permit the release actor to update release/update-version, and permit
the release actor to create tags/releases. Keep owner reviews and branch protection.
Require the Linux/Windows validation checks; allow merge commits for main
promotion. The release workflow itself creates no source commits on main/develop.
No repository settings or secret values were changed. Live token scopes, workflow
delivery, PR creation and publication require the owner's next reviewed push;
no workflow was dispatched here.

## Remaining non-Rust tooling

Python release extraction/tests and Python installer tests were removed after
their behavior passed in Rust. Existing home/config migration validation remains
in Rust. CI-critical release and bootstrap checks need no Python.

- install.sh and install.ps1 remain bootstrap download/checksum/install/PATH
  wrappers: no Rust binary is available until download and validation finish.
  Runtime home resolution/migration is already Rust. End users need no Rust SDK.
- scripts/migration_inference_trial.py remains an optional standard-library live
  model harness, outside CI. It orchestrates an existing binary, HTTP request and
  isolated fixtures; it contains no release/version generation. Rewriting this
  opt-in hardware helper adds little benefit to this focused release change.
- scripts/tui-review/render.py remains a Pillow renderer of Rust TestBackend
  exports; it depends on Windows fonts for optional PNG visual review. Assertions
  on TUI cells are already Rust. It is not a terminal-launch or quality test.
- demos/record.ps1, demos/overview/driver.ps1, demos/tts/driver.ps1 and
  demos/common/driver_helpers.ps1 remain existing VHS recording/request/report
  orchestration, with no release generation. Their GUI/media tooling is outside
  this change. cog.toml/commit-msg retain optional conventional commit validation;
  cliff.toml is historical configuration, superseded by xtask for releases.
- GitHub Actions YAML, third-party Actions, git/gh transport and packaging shell
  commands remain their native integration formats. No AI service is called.

## Local evidence

Windows: application suite 182 passed, five ignored; nine release utility tests
and two installer acceptance tests passed. Installer acceptance covers native
PowerShell 5 and Git Bash, repeat installs, spaces/apostrophes, checksums, explicit
overrides, conflict refusal, Kit sentinels, PATH deduplication and POSIX rollback.
Fresh migrated CUDA text inference returned HTTP 200 and READY; original config,
environment, model weights and Kit sentinels remained intact. The actual TUI
launched in a terminal and q exited successfully, restoring the terminal.

Native Linux/macOS/ARM execution, live GitHub events/token policies and live Omni
audio/vision/quality/co-residency remain unverified. Kit's own installer was not
run; sentinel preservation verifies only Infer's side of coexistence.
