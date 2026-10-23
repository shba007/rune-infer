use semver::Version;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, path::Path, process::Command};
use toml_edit::{DocumentMut, value};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
    pub source_head: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub group: &'static str,
    pub text: String,
    pub breaking: bool,
}

fn private_or_placeholder(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let words: Vec<_> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    words.iter().any(|w| {
        matches!(
            *w,
            "todo" | "tbd" | "placeholder" | "unreleased" | "cycle" | "milestone"
        )
    }) || words.windows(2).any(|pair| {
        matches!(pair[0], "m" | "k" | "l" | "r" | "inf" | "kf" | "wp" | "if")
            && pair[1].chars().all(|c| c.is_ascii_digit())
    }) || words.iter().any(|w| {
        let (prefix, number) = w.split_at(w.find(|c: char| c.is_ascii_digit()).unwrap_or(w.len()));
        !number.is_empty()
            && number.chars().all(|c| c.is_ascii_digit())
            && matches!(prefix, "m" | "k" | "l" | "r" | "inf" | "kf" | "wp" | "if")
    }) || lower.contains("ci(release):")
        || lower.contains("update version")
        || lower.contains("bump version")
        || lower.contains("version bump")
        || lower.contains("coming soon")
}

/// Conventional Commit parsing is supplied by git-conventional, not an AI service.
pub fn entries(messages: &[String]) -> Vec<Entry> {
    let mut seen = BTreeSet::new();
    messages
        .iter()
        .filter_map(|message| {
            let commit = git_conventional::Commit::parse(message).ok()?;
            let kind = commit.type_().to_string();
            let scope = commit.scope().map(|s| s.to_string());
            if (scope.as_deref() == Some("release")
                && commit.description().starts_with("publish v"))
                || commit
                    .description()
                    .to_ascii_lowercase()
                    .contains("apply code fixes")
                || private_or_placeholder(commit.description())
            {
                return None;
            }
            let group = if commit.breaking() {
                "Compatibility"
            } else {
                match kind.as_str() {
                    "feat" => "Features",
                    "fix" => "Fixes",
                    "perf" => "Performance",
                    _ => "Maintenance",
                }
            };
            let description = commit.description().replace(['\r', '\n'], " ");
            let text = if let Some(scope) = scope.filter(|s| !private_or_placeholder(s)) {
                format!("{scope}: {description}")
            } else {
                description
            };
            if !seen.insert(text.clone()) {
                return None;
            }
            Some(Entry {
                group,
                text,
                breaking: commit.breaking(),
            })
        })
        .collect()
}

fn sections(changelog: &str) -> Vec<(Version, usize, usize)> {
    let mut offset = 0;
    let mut result = Vec::new();
    for line in changelog.split_inclusive('\n') {
        if let Some(title) = line.trim_end().strip_prefix("## ")
            && let Ok(version) = Version::parse(if let Some(bracketed) = title.strip_prefix('[') {
                bracketed.split(']').next().unwrap_or("")
            } else {
                title.strip_prefix('v').unwrap_or(title)
            })
        {
            result.push((version, offset, offset + line.len()));
        }
        offset += line.len();
    }
    result
}

pub fn release_notes(changelog: &str, version: &Version) -> Result<String> {
    // Git can check out CRLF on Windows; release prose must compare identically
    // with the LF body published by the Linux preparation/publication jobs.
    let normalized = changelog.replace("\r\n", "\n");
    let changelog = normalized.as_str();
    let parts = sections(changelog);
    if parts.first().map(|p| &p.0) != Some(version)
        || parts.iter().filter(|p| &p.0 == version).count() != 1
    {
        return Err(
            format!("First CHANGELOG version section must uniquely match {version}").into(),
        );
    }
    let notes = changelog[parts[0].2..parts.get(1).map_or(changelog.len(), |p| p.1)].trim();
    let substantive = notes
        .lines()
        .any(|l| l.trim_start().starts_with("- ") && l.trim().len() > 2);
    if !substantive || private_or_placeholder(notes) {
        return Err(format!(
            "CHANGELOG {version} has empty, placeholder or internal planning notes"
        )
        .into());
    }
    Ok(format!(
        "{}\n",
        changelog[parts[0].1..parts.get(1).map_or(changelog.len(), |p| p.1)].trim_end()
    ))
}

#[derive(Debug)]
pub struct Plan {
    pub version: Version,
    pub changelog: String,
    pub notes: String,
    pub state: State,
}

pub fn plan(
    published: &Version,
    current: &Version,
    changelog: &str,
    source_head: &str,
    changes: &[Entry],
) -> Result<Option<Plan>> {
    if changes.is_empty() {
        return Ok(None);
    }
    if current < published {
        return Err("Manifest version is older than the latest published tag".into());
    }
    let version = {
        let mut next = published.clone();
        // Infer's existing cliff rules: features always minor, breaking always major.
        if changes.iter().any(|c| c.breaking) {
            next.major += 1;
            next.minor = 0;
            next.patch = 0;
        } else if changes.iter().any(|c| c.group == "Features") {
            next.minor += 1;
            next.patch = 0;
        } else {
            next.patch += 1;
        }
        std::cmp::max(next, current.clone())
    };
    let pending = current > published;
    let old_notes = if pending {
        release_notes(changelog, current)?
            .split_once('\n')
            .ok_or("Missing section body")?
            .1
            .trim()
            .to_owned()
    } else {
        String::new()
    };
    let mut additions = String::new();
    for group in [
        "Features",
        "Fixes",
        "Performance",
        "Maintenance",
        "Compatibility",
    ] {
        let lines: Vec<_> = changes
            .iter()
            .filter(|c| c.group == group)
            .filter(|c| {
                !old_notes
                    .lines()
                    .any(|line| line == format!("- {}", c.text))
            })
            .map(|c| format!("- {}\n", c.text))
            .collect();
        if !lines.is_empty() {
            additions.push_str(&format!("### {group}\n\n{}\n", lines.concat()));
        }
    }
    let notes = if old_notes.is_empty() {
        additions.trim_end().to_owned()
    } else if additions.is_empty() {
        old_notes.trim_end().to_owned()
    } else {
        format!("{}\n\n{}", old_notes.trim_end(), additions.trim_end())
    };
    let parts = sections(changelog);
    let updated = if pending {
        let first = parts.first().ok_or("Missing pending version section")?;
        let end = parts.get(1).map_or(changelog.len(), |p| p.1);
        format!(
            "{}## [{version}]\n\n{notes}\n\n{}",
            &changelog[..first.1],
            &changelog[end..]
        )
    } else {
        let at = parts.first().map_or(changelog.len(), |p| p.1);
        format!(
            "{}## [{version}]\n\n{notes}\n\n{}",
            &changelog[..at],
            &changelog[at..]
        )
    };
    let notes = release_notes(&updated, &version)?;
    Ok(Some(Plan {
        version: version.clone(),
        changelog: updated,
        notes,
        state: State {
            source_head: source_head.into(),
            version: version.to_string(),
        },
    }))
}

pub fn git(root: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git").current_dir(root).args(args).output()?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(out.stdout)?)
}

pub fn version(root: &Path) -> Result<Version> {
    let cargo = fs::read_to_string(root.join("Cargo.toml"))?.parse::<DocumentMut>()?;
    Ok(Version::parse(
        cargo["package"]["version"]
            .as_str()
            .ok_or("Missing Cargo package version")?,
    )?)
}

pub fn repository_plan(root: &Path) -> Result<Option<Plan>> {
    let tags = git(root, &["tag", "--list", "v*"])?;
    let published = tags
        .lines()
        .filter_map(|tag| Version::parse(tag.strip_prefix('v')?).ok())
        .filter(|v| v.pre.is_empty())
        .max()
        .ok_or("No release tags found; fetch complete history and tags")?;
    let current = version(root)?;
    let head = git(root, &["rev-parse", "HEAD"])?.trim().to_owned();
    let state_path = root.join(".release/prepared.json");
    let state: Option<State> = if state_path.exists() {
        Some(serde_json::from_slice(&fs::read(state_path)?)?)
    } else {
        None
    };
    let base = history_base(&published, &current, state.as_ref())?;
    if base == format!("v{published}") {
        // Publication tags the main promotion merge, which need not itself be
        // an ancestor of develop. tag..HEAD still excludes all reviewed commits
        // reachable through that merge's develop parent. Require related history.
        git(root, &["merge-base", &base, "HEAD"])?;
    } else {
        git(root, &["merge-base", "--is-ancestor", &base, "HEAD"])?;
    }
    let history = git(
        root,
        &[
            "log",
            "--no-merges",
            "--reverse",
            "--format=%B%x00",
            &format!("{base}..HEAD"),
        ],
    )?;
    let messages = history
        .split('\0')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    plan(
        &published,
        &current,
        &fs::read_to_string(root.join("CHANGELOG.md"))?,
        &head,
        &entries(&messages),
    )
}

pub fn history_base(
    published: &Version,
    current: &Version,
    state: Option<&State>,
) -> Result<String> {
    if let Some(state) = state {
        if Version::parse(&state.version)? <= *published {
            return Ok(format!("v{published}"));
        }
        if state.version != current.to_string()
            || state.source_head.len() != 40
            || !state.source_head.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Err("Invalid pending release state".into());
        }
        Ok(state.source_head.clone())
    } else {
        Ok(format!("v{published}"))
    }
}

/// Only changes release-owned files. Does not stage, commit, push or invoke Git mutations.
pub fn apply(root: &Path, plan: &Plan) -> Result<()> {
    let mut cargo = fs::read_to_string(root.join("Cargo.toml"))?.parse::<DocumentMut>()?;
    cargo["package"]["version"] = value(plan.version.to_string());
    let mut lock = fs::read_to_string(root.join("Cargo.lock"))?.parse::<DocumentMut>()?;
    let packages = lock["package"]
        .as_array_of_tables_mut()
        .ok_or("Cargo.lock package list missing")?;
    let package_lock = packages
        .iter_mut()
        .find(|p| p["name"].as_str() == Some("rune-infer"))
        .ok_or("Cargo.lock Infer package missing")?;
    package_lock["version"] = value(plan.version.to_string());
    fs::write(root.join("Cargo.toml"), cargo.to_string())?;
    fs::write(root.join("Cargo.lock"), lock.to_string())?;
    fs::write(root.join("CHANGELOG.md"), &plan.changelog)?;
    fs::create_dir_all(root.join(".release"))?;
    fs::write(
        root.join(".release/prepared.json"),
        format!("{}\n", serde_json::to_string_pretty(&plan.state)?),
    )?;
    Ok(())
}

pub fn validate_versions(root: &Path) -> Result<()> {
    let version = version(root)?.to_string();
    let cargo = fs::read_to_string(root.join("Cargo.toml"))?.parse::<DocumentMut>()?;
    let lock = fs::read_to_string(root.join("Cargo.lock"))?.parse::<DocumentMut>()?;
    if cargo["package"]["version"].as_str() != Some(&version)
        || !lock["package"]
            .as_array_of_tables()
            .ok_or("Missing lock packages")?
            .iter()
            .any(|p| {
                p["name"].as_str() == Some("rune-infer") && p["version"].as_str() == Some(&version)
            })
    {
        return Err("Cargo.toml and Cargo.lock versions must agree".into());
    }
    release_notes(
        &fs::read_to_string(root.join("CHANGELOG.md"))?,
        &Version::parse(&version)?,
    )?;
    Ok(())
}

pub fn validate_event(root: &Path, event: &serde_json::Value) -> Result<()> {
    validate_versions(root)?;
    let version = version(root)?;
    let notes = release_notes(&fs::read_to_string(root.join("CHANGELOG.md"))?, &version)?;
    if event["release"]["tag_name"] != format!("v{version}")
        || event["release"]["body"].as_str().map(str::trim) != Some(notes.trim())
    {
        return Err("Published version/body does not match the prepared CHANGELOG section".into());
    }
    Ok(())
}

/// Main promotion must preserve reviewed history and carry an explicit owner marker.
pub fn validate_promotion(root: &Path) -> Result<()> {
    validate_versions(root)?;
    let version = version(root)?;
    let state: State = serde_json::from_slice(&fs::read(root.join(".release/prepared.json"))?)?;
    if state.version != version.to_string()
        || state.source_head.len() != 40
        || !state.source_head.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Err("Missing or mismatched prepared release state".into());
    }
    git(
        root,
        &["merge-base", "--is-ancestor", &state.source_head, "HEAD"],
    )?;
    if git(root, &["show", "-s", "--format=%P", "HEAD"])?
        .split_whitespace()
        .count()
        != 2
        || git(root, &["show", "-s", "--format=%s", "HEAD"])?.trim()
            != format!("ci(release): publish v{version}")
    {
        return Err(
            "Promote develop with a merge commit titled ci(release): publish vX.Y.Z".into(),
        );
    }
    Ok(())
}
