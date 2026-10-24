//! Read-only integration with pinned Cocogitto. Only Cocogitto renders notes.
use cocogitto::{
    CocoGitto,
    conventional::{
        changelog::{ReleaseType, template::Template},
        commit::Commit as CogCommit,
        version::Increment,
    },
    git::{oid::OidOf, tag::Tag},
};
use semver::Version;
use std::{path::Path, sync::Once};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[allow(dead_code)]
pub const GENERATOR: &str = "cocogitto@7.0.0";
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Signature {
    pub name: Option<String>,
    pub timestamp: i64,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Commit {
    pub id: String,
    pub message: String,
    pub author: Signature,
}
impl Commit {
    pub fn new(id: String, message: String) -> Self {
        Self {
            id,
            message,
            author: Signature::default(),
        }
    }
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Release {
    pub version: Option<String>,
    pub commits: Vec<Commit>,
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Generated {
    pub releases: Vec<Release>,
    pub markdown: String,
}
fn initialize() -> Result<&'static cocogitto::settings::Settings> {
    static INIT: Once = Once::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    INIT.call_once(|| {
        cocogitto::set_config_path(root.join("cog.toml").to_string_lossy().into_owned())
    });
    // Validate configuration explicitly; do not accept Cocogitto's default fallback.
    CocoGitto::get_at(root.to_path_buf()).map_err(|e| e.to_string())?;
    let reviewed =
        cocogitto::settings::Settings::try_from(std::fs::read_to_string(root.join("cog.toml"))?)?;
    if *cocogitto::SETTINGS != reviewed {
        return Err(
            "Cocogitto did not load the reviewed configuration; run from the repository root"
                .into(),
        );
    }
    Ok(&cocogitto::SETTINGS)
}
fn excluded(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    let words: Vec<_> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    words
        .iter()
        .any(|w| matches!(*w, "todo" | "tbd" | "placeholder" | "cycle" | "milestone"))
        || words.iter().any(|w| {
            let (prefix, number) =
                w.split_at(w.find(|c: char| c.is_ascii_digit()).unwrap_or(w.len()));
            matches!(
                prefix,
                "m" | "k" | "l" | "r" | "inf" | "if" | "kf" | "kit" | "wp"
            ) && !number.is_empty()
                && number.chars().all(|c| c.is_ascii_digit())
        })
        || words.windows(2).any(|w| {
            matches!(
                w[0],
                "m" | "k" | "l" | "r" | "inf" | "if" | "kf" | "kit" | "wp"
            ) && w[1].chars().all(|c| c.is_ascii_digit())
        })
        || [
            "apply code fixes",
            "autoformat",
            "bump version",
            "version bump",
            "update version",
            "bump packages",
            "coming soon",
        ]
        .iter()
        .any(|w| lower.contains(w))
        || lower.starts_with("chore(registry):")
        || (lower.starts_with("ci(release):")
            && (lower.contains("publish v") || lower.contains("prepare release")))
}
fn parsed(commit: &Commit) -> Option<CogCommit> {
    let conventional = conventional_commit_parser::parse(&commit.message).ok()?;
    Some(CogCommit {
        oid: commit.id.clone(),
        conventional,
        author: commit.author.name.clone().unwrap_or_default(),
        date: chrono::DateTime::from_timestamp(commit.author.timestamp, 0)?.naive_utc(),
    })
}
pub fn generate(
    base: &str,
    commits: Vec<Commit>,
    timestamp: i64,
    package: Option<&str>,
) -> Result<Generated> {
    initialize()?;
    let commits: Vec<_> = commits
        .into_iter()
        .filter(|c| !excluded(&c.message) && parsed(c).is_some())
        .collect();
    if commits.is_empty() {
        return Ok(Generated {
            releases: vec![Release {
                version: None,
                commits,
            }],
            markdown: String::new(),
        });
    }
    let previous = Version::parse(base.trim_start_matches('v'))?;
    let mut semantic: Vec<_> = commits.iter().filter_map(parsed).collect();
    for commit in &mut semantic {
        // Cocogitto exposes pre-1.0 breaking fixes as patch. Preserve Rune's
        // public policy by classifying them as features for bumping only.
        if previous.major == 0 && commit.conventional.is_breaking_change {
            commit.conventional.commit_type = "feat".into();
        } else if !cocogitto::COMMITS_METADATA.contains_key(&commit.conventional.commit_type) {
            commit.conventional.commit_type = "chore".into();
        }
    }
    let tag = Tag {
        package: None,
        prefix: Some("v".into()),
        version: previous.clone(),
        oid: None,
    };
    let increment = tag.version_increment_from_commit_history(&semantic)?;
    let mut next = previous;
    match increment {
        Increment::Major => {
            next.major += 1;
            next.minor = 0;
            next.patch = 0;
        }
        Increment::Minor => {
            next.minor += 1;
            next.patch = 0;
        }
        Increment::Patch => next.patch += 1,
        Increment::NoBump => {
            return Err("Eligible commits did not produce a Cocogitto version increment".into());
        }
    }
    next.pre = semver::Prerelease::EMPTY;
    next.build = semver::BuildMetadata::EMPTY;
    let markdown = render_at(Some(base), &next.to_string(), &commits, timestamp, package)?;
    Ok(Generated {
        releases: vec![Release {
            version: Some(next.to_string()),
            commits,
        }],
        markdown,
    })
}
#[allow(dead_code)]
pub fn render_at(
    base: Option<&str>,
    version: &str,
    commits: &[Commit],
    timestamp: i64,
    package: Option<&str>,
) -> Result<String> {
    let settings = initialize()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let coco = CocoGitto::get_at(root.to_path_buf()).map_err(|e| e.to_string())?;
    // The public API returns the release type. Read one commit to obtain it,
    // then supply the already-selected, reviewed history without touching Git.
    let mut release = coco
        .get_changelog("HEAD~1..HEAD", false)
        .map_err(|e| e.to_string())?;
    release.previous = None;
    release.commits = commits.iter().filter_map(parsed).map(Into::into).collect();
    release.date = chrono::DateTime::from_timestamp(timestamp, 0)
        .ok_or("Invalid release timestamp")?
        .naive_utc();
    let tag = |raw: &str| -> Result<Tag> {
        Ok(Tag {
            package: package.map(str::to_owned),
            prefix: Some("v".into()),
            version: Version::parse(raw.trim_start_matches('v'))?,
            oid: None,
        })
    };
    release.version = OidOf::Tag(tag(version)?);
    release.from = OidOf::Tag(tag(base.unwrap_or(version))?);
    let configured = settings
        .changelog
        .template
        .as_deref()
        .ok_or("Missing Cocogitto template")?;
    let template_path = root.join(configured);
    let template = Template::from_arg(
        &template_path.to_string_lossy(),
        settings.get_template_context(),
        false,
    )?;
    let markdown = release.into_markdown(template, ReleaseType::Standard)?;
    Ok(markdown.replace("\r\n", "\n").trim().to_string() + "\n")
}
