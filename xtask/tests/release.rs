use infer_xtask::{
    State, apply, entries, history_base, plan, release_notes, validate_event, validate_versions,
};
use semver::Version;
use std::fs;

fn v(s: &str) -> Version {
    Version::parse(s).unwrap()
}
fn history() -> String {
    "# Changelog\n\n## v0.0.23\n\n- Previous notes, preserved exactly.\n\n## v0.0.22\n\n- Historical Cycle 2 release.\n".into()
}
fn messages(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).into()).collect()
}

#[test]
fn deterministic_preparation_preserves_history_and_extracts_exact_notes() {
    let old = history();
    let changes = entries(&messages(&[
        "fix(kit): detect incompatible gateways",
        "feat(tools): expose bounded directory listings",
    ]));
    let first = plan(&v("0.0.23"), &v("0.0.23"), &old, "source", &changes)
        .unwrap()
        .unwrap();
    let second = plan(&v("0.0.23"), &v("0.0.23"), &old, "source", &changes)
        .unwrap()
        .unwrap();
    assert_eq!(first.version, v("0.1.0"));
    assert_eq!(first.changelog, second.changelog);
    assert!(
        first
            .changelog
            .ends_with(&old[old.find("## v0.0.23").unwrap()..])
    );
    assert_eq!(
        release_notes(&first.changelog, &first.version).unwrap(),
        first.notes
    );
    assert!(
        plan(
            &v("0.0.23"),
            &first.version,
            &first.changelog,
            "source",
            &[]
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn pending_version_is_reused_and_entries_are_not_duplicated() {
    let changes = entries(&messages(&["fix(kit): detect incompatible gateways"]));
    let first = plan(&v("0.0.23"), &v("0.0.23"), &history(), "source", &changes)
        .unwrap()
        .unwrap();
    let second = plan(
        &v("0.0.23"),
        &first.version,
        &first.changelog,
        "later",
        &changes,
    )
    .unwrap()
    .unwrap();
    assert_eq!(first.changelog, second.changelog);
    assert_eq!(first.version, second.version);
    let extra = entries(&messages(&["fix(ui): explain unavailable plugins"]));
    let third = plan(
        &v("0.0.23"),
        &first.version,
        &first.changelog,
        "new-source",
        &extra,
    )
    .unwrap()
    .unwrap();
    assert_eq!(third.version, first.version);
    assert!(third.notes.contains("unavailable plugins"));
    assert_eq!(third.changelog.matches("## [0.0.24]").count(), 1);
}

#[test]
fn routine_bot_merge_and_internal_messages_do_not_prepare_releases() {
    let changes = entries(&messages(&[
        "ci(release): update version",
        "chore: apply code fixes [skip ci]",
        "chore: update version to 0.0.23",
        "Merge pull request #12 from release/update-version",
        "feat(kit): finish Cycle 3",
        "fix: finish M15 checks",
        "fix: complete INF-7 rollout",
        "not a conventional commit",
        "fix: TBD",
        "fix: update version",
    ]));
    assert!(changes.is_empty());
    assert!(
        plan(&v("0.0.23"), &v("0.0.23"), &history(), "head", &changes)
            .unwrap()
            .is_none()
    );
}

#[test]
fn every_semantic_type_can_prepare_a_release() {
    for kind in [
        "feat", "fix", "perf", "docs", "refactor", "style", "test", "build", "ci", "chore",
        "revert", "custom",
    ] {
        let changes = entries(&[format!("{kind}: improve {kind} behavior")]);
        assert_eq!(changes.len(), 1, "{kind} was excluded");
        let prepared = plan(&v("0.0.23"), &v("0.0.23"), &history(), "head", &changes)
            .unwrap()
            .unwrap();
        assert_eq!(
            prepared.version,
            v(if kind == "feat" { "0.1.0" } else { "0.0.24" })
        );
        assert!(
            prepared
                .notes
                .contains(&format!("- improve {kind} behavior"))
        );
    }
    let changes = entries(&messages(&["refactor: remove release notes test file"]));
    assert!(
        plan(&v("0.0.23"), &v("0.0.23"), &history(), "head", &changes)
            .unwrap()
            .unwrap()
            .notes
            .contains("remove release notes test file")
    );
}

#[test]
fn consumed_source_head_prevents_preparation_loops_after_merge() {
    let state = State {
        version: "0.0.24".into(),
        source_head: "a".repeat(40),
    };
    assert_eq!(
        history_base(&v("0.0.23"), &v("0.0.24"), Some(&state)).unwrap(),
        state.source_head
    );
    assert_eq!(
        history_base(&v("0.0.24"), &v("0.0.24"), Some(&state)).unwrap(),
        "v0.0.24"
    );
    assert!(history_base(&v("0.0.23"), &v("0.0.25"), Some(&state)).is_err());
    let invalid = State {
        source_head: "not-a-commit".into(),
        ..state
    };
    assert!(history_base(&v("0.0.23"), &v("0.0.24"), Some(&invalid)).is_err());
}

#[test]
fn conventional_commit_breaking_and_stable_version_rules() {
    for (base, message, expected) in [
        ("0.0.23", "feat!: require a compatible gateway", "1.0.0"),
        ("1.2.3", "feat: add bounded reads", "1.3.0"),
        ("1.2.3", "fix: explain errors", "1.2.4"),
        (
            "1.2.3",
            "fix: change transport\n\nBREAKING CHANGE: old transport is unsupported",
            "2.0.0",
        ),
    ] {
        let log = format!("# Changelog\n\n## v{base}\n\n- Old notes.\n");
        let plan = plan(
            &v(base),
            &v(base),
            &log,
            "head",
            &entries(&messages(&[message])),
        )
        .unwrap()
        .unwrap();
        assert_eq!(plan.version, v(expected));
    }
}

#[test]
fn substantive_release_fixes_are_retained_and_pending_breaking_changes_advance_major() {
    let changes = entries(&messages(&[
        "fix(release): preserve installer notes",
        "ci(release): repair artifact delivery",
        "fix: apply code fixes [skip ci]",
    ]));
    assert_eq!(changes.len(), 2);
    assert!(changes[0].text.contains("preserve installer notes"));
    let pending = "# Changelog\n\n## [0.0.24]\n\n- An unreleased fix.\n\n## v0.0.23\n\n- Old published fix.\n";
    // Avoid a placeholder word in substantive release prose.
    let pending = pending.replace("An unreleased fix", "An existing pending fix");
    let change = entries(&messages(&["feat!: require a new transport"]));
    let prepared = plan(&v("0.0.23"), &v("0.0.24"), &pending, "head", &change)
        .unwrap()
        .unwrap();
    assert_eq!(prepared.version, v("1.0.0"));
    assert!(prepared.notes.contains("existing pending fix"));
}

#[test]
fn notes_fail_on_missing_mismatched_empty_duplicate_and_placeholder_sections() {
    for log in [
        "## v0.0.22\n- Old.",
        "## [0.0.24]\n- Future.\n## v0.0.23\n- Current.",
        "## v0.0.23\n\n",
        "## v0.0.23\n### Fixes",
        "## v0.0.23\n- TODO",
        "## [0.0.23] - 2026-10-23\n- Coming soon",
        "## v0.0.23\n- Version bump to 0.0.23",
        "## v0.0.23\n- Cycle 3 changes",
        "## v0.0.23\n- ci(release): update version",
        "## v0.0.23\n- Good.\n## v0.0.23\n- Duplicate.",
    ] {
        assert!(release_notes(log, &v("0.0.23")).is_err(), "{log}");
    }
    assert_eq!(
        release_notes(
            "# Changelog\r\n\r\n## v0.0.23\r\n\r\n- Actual fix.\r\n",
            &v("0.0.23")
        )
        .unwrap(),
        "## v0.0.23\n\n- Actual fix.\n"
    );
}

#[test]
fn apply_synchronizes_manifests_and_lock_without_touching_unrelated_files() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"rune-infer\"\nversion = \"0.0.23\"\n",
    )
    .unwrap();
    fs::write(root.join("Cargo.lock"), "version = 4\n[[package]]\nname = \"rune-infer\"\nversion = \"0.0.23\"\n[[package]]\nname = \"other\"\nversion = \"1.0.0\"\n").unwrap();
    fs::write(root.join("CHANGELOG.md"), history()).unwrap();
    fs::write(root.join("unrelated.txt"), "owner content").unwrap();
    let prepared = plan(
        &v("0.0.23"),
        &v("0.0.23"),
        &history(),
        "source-head",
        &entries(&messages(&["fix: explain errors"])),
    )
    .unwrap()
    .unwrap();
    apply(root, &prepared).unwrap();
    validate_versions(root).unwrap();
    let valid_event = serde_json::json!({"release":{"tag_name":"v0.0.24","body":prepared.notes}});
    validate_event(root, &valid_event).unwrap();
    fs::write(
        root.join("CHANGELOG.md"),
        prepared.changelog.replace('\n', "\r\n"),
    )
    .unwrap();
    validate_event(root, &valid_event).unwrap();
    fs::write(root.join("CHANGELOG.md"), &prepared.changelog).unwrap();
    for invalid in [
        serde_json::json!({"release":{"tag_name":"v0.0.23","body":prepared.notes}}),
        serde_json::json!({"release":{"tag_name":"v0.0.24","body":"Regenerated notes from history"}}),
    ] {
        assert!(validate_event(root, &invalid).is_err());
    }
    let lock = fs::read_to_string(root.join("Cargo.lock")).unwrap();
    assert!(lock.contains("version = \"0.0.24\""));
    assert!(lock.contains("version = \"1.0.0\""));
    assert_eq!(
        fs::read_to_string(root.join("unrelated.txt")).unwrap(),
        "owner content"
    );
    let before = fs::read(root.join("CHANGELOG.md")).unwrap();
    apply(root, &prepared).unwrap();
    assert_eq!(before, fs::read(root.join("CHANGELOG.md")).unwrap());
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"rune-infer\"\nversion = \"0.0.1\"\n",
    )
    .unwrap();
    assert!(validate_versions(root).is_err());
}
