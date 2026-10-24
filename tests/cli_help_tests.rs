use std::process::Command;

#[test]
fn bare_invocation_shows_help_without_resolving_configuration() {
    let home = std::env::temp_dir().join(format!("rune-infer-help-{}", std::process::id()));
    assert!(!home.exists());
    let output = Command::new(env!("CARGO_BIN_EXE_rune-infer"))
        .current_dir(std::env::temp_dir())
        .env("RUNE_HOME", &home)
        .output()
        .expect("run CLI");
    assert!(output.status.success(), "{:?}", output);
    let help = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "model server",
        "Usage:",
        "Commands:",
        "serve",
        "setup",
        "migrate",
    ] {
        assert!(help.contains(expected), "missing {expected}: {help}");
    }
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
    assert!(!home.exists(), "help must not create an Infer home");
}
