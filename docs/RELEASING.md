# Automated preparation and publication

Agents implement/test locally and leave changes unstaged/uncommitted. The owner
commits/pushes development work and reviews/merges release PRs. GitHub Actions may
prepare release files, push release/update-version and publish after promotion.
No AI service generates release prose. Agents must not author or edit CHANGELOG
entries, including the pending version; only generated temporary fixtures and
read-only previews are used locally.

## Pipeline and owner actions

1. Owner pushes reviewed semantic commits to develop. Rust CI runs Linux/Windows
   build/tests/Clippy/format, including migration and bootstrap acceptance. Only
   develop pushes trigger integration/preparation, including merge/release commits.
   PR events and release-branch pushes do not duplicate the pipeline. A lightweight
   Rust classification job skips the full matrix/preparation for validated release
   metadata-only merges into develop. It checks changed paths, version-only
   manifest/lock edits and prepared metadata; source/dependency changes or invalid
   release metadata still run normal CI, regardless of subject.
2. After both validation jobs pass, the serialized preparation job checks out
   develop and verifies its SHA equals the tested source. Pinned git-cliff generates
   the version and substantive changelog from reviewed commits. Rust synchronizes
   Cargo.toml/Cargo.lock and .release/prepared.json. Generated files are validated
   in this same run, including build/tests/Clippy/format, before PR creation. A final
   fetch rejects a moved develop head; publication independently rejects unprepared
   semantic commits after the recorded source, closing the remaining race.
3. Actions creates/updates one release/update-version → develop PR, with commit and
   title `ci(release): update version`. Only release-owned files are committed. No
   new releasable commits means no PR. Re-runs produce the same pending version and
   section, rather than repeatedly bumping or appending duplicate sections.
4. Owner reviews the generated version, notes and successful original develop run.
   Merge or squash the preparation PR into develop. Then promote develop → main
   with a MERGE COMMIT, preserving reviewed ancestry; do not squash/rebase promotion.
   There is **no required main commit title**. Merging into develop alone does not
   publish. The previous title gate skipped main run 40004416448 when the subject
   was `ci(release): update version`; that gate is removed from workflow and Rust.
5. Every main push runs release eligibility validation, regardless of title or
   parent count. It validates synchronized manifest/lock versions and metadata.
   - Current version already tagged: successful explicit no-pending outcome, no
     credential needed and no tag/release replacement.
   - Unpublished version: require matching prepared metadata, valid ancestor source,
     exact latest nonempty changelog section and matching stored note digest when
     available. Reject unprepared semantic changes after that source and versions
     differing from git-cliff's calculation for the complete pending history.
   - Malformed/decreasing versions, missing/mismatched pending metadata, stale source
     and invalid notes fail clearly. They are not silent job skips.
6. A valid unpublished pending release is published with GH_TOKEN at the main SHA,
   using the exact prepared section as the body. Release creation triggers the
   existing release.published artifact workflow. Each artifact job checks out the
   release tag and validates the event tag/body with the same note extractor before
   uploading archives, checksums and bootstrap installers. Existing assets/tags
   are never replaced. Owner verifies actual event delivery and downloads.

## Package and version policy

`git-cliff-core =2.14.2` is pinned in xtask/Cargo.toml and Cargo.lock. Active
cliff.toml configures its parser, template, grouping and bump behavior. There is
no custom version-increment algorithm or handwritten changelog renderer. Features
advance minor, fix/perf/other semantic maintenance advances patch, and breaking
changes advance minor below 1.0 or major at/above 1.0. Breaking classification wins
regardless of ordinary type. All semantic types, including docs/refactor/style/
test/build/ci/chore/revert/custom, remain eligible. Routine release/autoformat
bookkeeping, invalid messages, placeholders and internal planning identifiers are
excluded by package configuration, not by an AI service.

The package template follows changelogen presentation: version heading, compare
link with actual previous/current tags, nonempty grouped sections (Enhancements,
Fixes, Documentation, Refactors, Performance, Tests, Build, CI, Chore), commit links,
PR references where present, explicit breaking information and locally known
contributors. Remote requests are disabled, so author names from Git supply
contributors without API-dependent nondeterminism. Historical sections remain
byte-for-byte intact. Preparation replaces only the unpublished section and never
regenerates published history.

Full tag-to-source pending history determines the greatest package bump and the
complete section. .release/prepared.json records consumed source/version and, for
new preparations, generator identity and SHA-256 of the exact note section. The
consumed source prevents preparation loops after merge/squash. A publication tag
on main's merge need not be an ancestor of develop: its reachability still excludes
already reviewed source from the next tag-to-develop range. Fetch full tags/history.
Legacy prepared metadata with only sourceHead/version remains accepted for the
existing 0.8.0 promotion, with ancestry/version/notes/package-calculation checks;
subsequent CI preparation adds generator/digest fields.

Publication extracts the exact matching prepared version section, including its
heading, with canonical LF and one final newline. Missing, duplicate, mismatched,
empty or placeholder notes fail. It does not render fresh history prose at merge.
Artifact event validation uses that identical extraction function. A valid existing
tag counts as already released; if a tag was manually created without a release,
the owner must inspect/reconcile that exceptional state. Automation never moves it.

## Local commands

```text
cargo build --locked
cargo test --locked --manifest-path xtask/Cargo.toml
cargo clippy --locked --manifest-path xtask/Cargo.toml --all-targets -- -D warnings
cargo run --locked --manifest-path xtask/Cargo.toml -- prepare
cargo run --locked --manifest-path xtask/Cargo.toml -- example
cargo run --locked --manifest-path xtask/Cargo.toml -- notes
cargo run --locked --manifest-path xtask/Cargo.toml -- verify-promotion
```

Prepare is a read-only dry run by default. --apply is reserved for authorized
CI/owner use; it writes release files only and never stages/commits/pushes. Tests
apply only inside temporary fixtures. Agents do not apply on this working repo.
Example uses package-generated fixture commits without making Git commits. The
current consumed source has only release bookkeeping afterward, so prepare reports
no new releasable changes until the owner commits the implementation. The existing
0.8.0 pending section was not edited by the agent. Review the next CI-generated PR
for current hidden-root migration descriptions and version before promotion.

## Credentials, settings and events

Use GITHUB_TOKEN (job Contents write/Pull requests write) for PR API operations,
and GH_TOKEN (PAT or refreshed GitHub App token with Contents write) for release
branch pushes and release publication. PR creation does not require the PAT to
have Pull requests write. Configure repository/org policy to permit Actions PRs,
the dedicated release branch, tags and release uploads; approve required token
repository access/SSO and any workflow-file permission needed by push ancestry.
Align branch protection with develop-only validation: a separate PR-head CI run
will not arrive. Keep owner review and allow merge commits for promotion.

[GitHub event rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
restrict downstream runs from GITHUB_TOKEN. GH_TOKEN release publication is used
so release.published starts Rust CD; artifact uploads use GITHUB_TOKEN because
no further workflow must be triggered. Bot PR event delivery is not relied upon:
generated-file validation runs before PR creation in the original develop run.
Missing credentials fail only when publication/preparation needs them; a main
no-pending result needs no publication secret. Never log secret values.

No workflow was dispatched, tag/release created, token inspected, or repository
setting changed locally. The owner must verify actual token permissions, the next
main validation/publication run and every platform artifact outcome. Re-running an
old workflow SHA still uses its old title-gated workflow; push the reviewed fix
through develop/preparation/promotion rather than assuming an old rerun uses it.

## Scope and remaining tooling

Bootstrap install.sh/install.ps1 stay usable without a Rust SDK; runtime home
resolution and migration are Rust. Optional Python live-model/visual helpers and
PowerShell/VHS demos remain outside CI release generation. They are not duplicate
version/changelog generators. Infer stays an HTTP inference server with its TUI;
Screen Insight remains Studio orchestration. No model/modality scope expanded.

Tests cover package-calculated pre/post-1.0 versions, all semantic types, exclusions,
linked/grouped output, repeat preparation, historical preservation, synchronized
manifests, exact artifact notes, no-pending versus invalid pending and title-free
eligibility. Windows/Git Bash bootstrap and Rust home migration fixtures cover
hidden-root setup, previous visible/older homes, repeats/conflicts and Kit data.
Native Linux/macOS/ARM and live GitHub publication remain owner/platform checks.
