use git_cliff_core::commit::Commit;
use infer_xtask::{
    GENERATOR, State, apply, generated_release, history_base, pending_release, plan, release_notes,
    validate_event, validate_versions,
};
use semver::Version;
use std::fs;
fn v(s: &str) -> Version {
    Version::parse(s).unwrap()
}
fn commits(messages: &[&str]) -> Vec<Commit<'static>> {
    messages
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut c = Commit::new(format!("{:040x}", i + 1), (*text).into());
            c.author.name = Some("Fixture contributor".into());
            c
        })
        .collect()
}
fn history() -> &'static str {
    "# Changelog\n\n## [0.7.0]\n\n- Historical Cycle 2 functionality.\n"
}
#[test]
fn package_calculates_explicit_feature_breaking_and_maintenance_versions() {
    for (base, message, expected) in [
        ("0.7.0", "fix: preserve user settings", "0.7.1"),
        ("0.7.0", "perf: reduce image copies", "0.7.1"),
        ("0.7.0", "feat: install into hidden user root", "0.8.0"),
        ("0.7.0", "fix!: change default installation root", "0.8.0"),
        ("1.2.3", "feat: add explicit server command", "1.3.0"),
        (
            "1.2.3",
            "fix: change configuration\n\nBREAKING CHANGE: explicit overrides are required",
            "2.0.0",
        ),
    ] {
        let (version, notes) = generated_release(&v(base), commits(&[message]))
            .unwrap()
            .unwrap();
        assert_eq!(version, v(expected));
        assert!(notes.contains(&format!("compare/v{base}...v{expected}")));
        assert!(notes.contains("/commit/0000000000000000000000000000000000000001"));
        assert!(notes.contains("Fixture contributor"));
        if message.contains('!') || message.contains("BREAKING CHANGE") {
            assert!(notes.contains("BREAKING"));
        }
    }
    let (version, _) = generated_release(
        &v("1.2.3"),
        commits(&[
            "fix: preserve settings",
            "feat: add image input",
            "refactor!: change configuration shape",
        ]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(version, v("2.0.0"));
}
#[test]
fn package_keeps_all_semantic_types_and_substantive_release_fixes() {
    for kind in [
        "docs", "refactor", "style", "test", "build", "ci", "chore", "revert", "custom",
    ] {
        let (version, notes) = generated_release(
            &v("0.7.0"),
            commits(&[&format!("{kind}: improve operator instructions")]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(version, v("0.7.1"));
        assert!(notes.contains("improve operator instructions"));
    }
    let (_, notes) = generated_release(
        &v("0.7.0"),
        commits(&["ci(release): repair publication eligibility (#42)"]),
    )
    .unwrap()
    .unwrap();
    assert!(notes.contains("/pull/42"));
    assert!(notes.contains("### CI"));
}
#[test]
fn package_excludes_private_invalid_and_routine_bookkeeping() {
    for message in [
        "ci(release): update version",
        "ci(release): publish v0.8.0",
        "chore: apply code fixes [skip ci]",
        "not a semantic subject",
        "feat: finish Cycle 3",
        "fix: finish INF-8",
        "fix: TODO",
        "fix: preserve settings\n\nMilestone 3 internal details",
    ] {
        assert!(
            generated_release(&v("0.7.0"), commits(&[message]))
                .unwrap()
                .is_none(),
            "{message}"
        );
    }
}
#[test]
fn package_output_is_repeatable_and_history_remains_unchanged() {
    let changes = commits(&[
        "feat(install): preserve shared product files",
        "fix: preserve legacy environment",
    ]);
    let first = plan(
        &v("0.7.0"),
        &v("0.7.0"),
        history(),
        &"a".repeat(40),
        changes.clone(),
    )
    .unwrap()
    .unwrap();
    let second = plan(
        &v("0.7.0"),
        &first.version,
        &first.changelog,
        &"a".repeat(40),
        changes,
    )
    .unwrap()
    .unwrap();
    assert_eq!(first.version, second.version);
    assert_eq!(first.changelog, second.changelog);
    assert!(
        first
            .changelog
            .ends_with(&history()[history().find("## [0.7.0]").unwrap()..])
    );
    assert_eq!(
        first.notes,
        release_notes(&first.changelog, &first.version).unwrap()
    );
    assert_eq!(first.state.generator.as_deref(), Some(GENERATOR));
    assert!(first.state.notes_sha256.is_some());
}
#[test]
fn state_prevents_bookkeeping_loops_and_resets_after_publication() {
    let state = State {
        source_head: "a".repeat(40),
        version: "0.8.0".into(),
        generator: None,
        notes_sha256: None,
    };
    assert_eq!(
        history_base(&v("0.7.0"), &v("0.8.0"), Some(&state)).unwrap(),
        state.source_head
    );
    assert_eq!(
        history_base(&v("0.8.0"), &v("0.8.0"), Some(&state)).unwrap(),
        "v0.8.0"
    );
    assert!(history_base(&v("0.7.0"), &v("0.9.0"), Some(&state)).is_err());
}
#[test]
fn publication_is_title_independent_and_distinguishes_noop_from_invalid() {
    let state = State {
        source_head: "a".repeat(40),
        version: "0.8.0".into(),
        generator: None,
        notes_sha256: None,
    };
    // No commit subject is an input: update-version, arbitrary merge titles and
    // direct main pushes all receive the same metadata/version validation.
    assert!(pending_release(&v("0.8.0"), &v("0.7.0"), Some(&state)).unwrap());
    assert!(!pending_release(&v("0.8.0"), &v("0.8.0"), Some(&state)).unwrap());
    assert!(!pending_release(&v("0.8.0"), &v("0.8.0"), None).unwrap());
    assert!(pending_release(&v("0.8.0"), &v("0.7.0"), None).is_err());
    assert!(pending_release(&v("0.7.0"), &v("0.8.0"), Some(&state)).is_err());
    assert!(pending_release(&v("0.9.0"), &v("0.7.0"), Some(&state)).is_err());
    let invalid = State {
        source_head: "invalid".into(),
        ..state
    };
    assert!(pending_release(&v("0.8.0"), &v("0.7.0"), Some(&invalid)).is_err());
}
#[test]
fn invalid_pending_and_non_increasing_versions_fail() {
    assert!(
        plan(
            &v("0.8.0"),
            &v("0.7.0"),
            history(),
            "head",
            commits(&["fix: preserve settings"])
        )
        .is_err()
    );
    assert!(
        plan(
            &v("0.7.0"),
            &v("0.9.0"),
            history(),
            "head",
            commits(&["fix: preserve settings"])
        )
        .is_err()
    );
}
#[test]
fn extractor_rejects_missing_mismatch_duplicate_empty_and_placeholder_notes() {
    for notes in [
        "## v0.6.0\n- Previous functionality.",
        "## v0.7.0\n",
        "## v0.7.0\n- TODO",
        "## v0.7.0\n- Cycle 3 changes",
        "## v0.7.0\n- Version bump to 0.7.0",
        "## v0.7.0\n- Real change.\n## v0.7.0\n- Duplicate.",
    ] {
        assert!(release_notes(notes, &v("0.7.0")).is_err());
    }
}
#[test]
fn apply_synchronizes_release_files_and_event_notes_exactly() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname=\"rune-infer\"\nversion=\"0.7.0\"\n",
    )
    .unwrap();
    fs::write(root.join("Cargo.lock"),"version=4\n[[package]]\nname=\"rune-infer\"\nversion=\"0.7.0\"\n[[package]]\nname=\"unrelated\"\nversion=\"1.2.3\"\n").unwrap();
    fs::write(root.join("keep.txt"), "owner data").unwrap();
    let prepared = plan(
        &v("0.7.0"),
        &v("0.7.0"),
        history(),
        &"a".repeat(40),
        commits(&["feat: preserve existing configuration"]),
    )
    .unwrap()
    .unwrap();
    apply(root, &prepared).unwrap();
    validate_versions(root).unwrap();
    let event = serde_json::json!({"release":{"tag_name":"v0.8.0","body":prepared.notes}});
    validate_event(root, &event).unwrap();
    fs::write(
        root.join("CHANGELOG.md"),
        prepared.changelog.replace('\n', "\r\n"),
    )
    .unwrap();
    validate_event(root, &event).unwrap();
    assert!(
        validate_event(
            root,
            &serde_json::json!({"release":{"tag_name":"v0.8.0","body":"different prose"}})
        )
        .is_err()
    );
    apply(root, &prepared).unwrap();
    let before = fs::read(root.join("CHANGELOG.md")).unwrap();
    apply(root, &prepared).unwrap();
    assert_eq!(before, fs::read(root.join("CHANGELOG.md")).unwrap());
    assert_eq!(
        fs::read_to_string(root.join("keep.txt")).unwrap(),
        "owner data"
    );
    assert!(
        fs::read_to_string(root.join("Cargo.lock"))
            .unwrap()
            .contains("1.2.3")
    );
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname=\"rune-infer\"\nversion=\"invalid-version\"\n",
    )
    .unwrap();
    assert!(validate_versions(root).is_err());
}

#[test]
fn prepared_note_receipt_rejects_post_preparation_changes() {
    let plan = plan(
        &v("0.7.0"),
        &v("0.7.0"),
        history(),
        &"a".repeat(40),
        commits(&["fix: preserve user configuration"]),
    )
    .unwrap()
    .unwrap();
    infer_xtask::validate_note_receipt(&plan.state, &plan.notes).unwrap();
    assert!(
        infer_xtask::validate_note_receipt(&plan.state, &format!("{}\nchanged", plan.notes))
            .is_err()
    );
    let invalid = State {
        generator: Some("unrecognized-generator".into()),
        ..plan.state
    };
    assert!(infer_xtask::validate_note_receipt(&invalid, &plan.notes).is_err());
}
