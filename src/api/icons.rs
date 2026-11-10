//! Provider and model-family icons served to local clients.
use axum::{
    extract::Path,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

const ICONS: &[(&str, &str)] = &[
    ("qwen", include_str!("../../public/icons/models/qwen.svg")),
    (
        "deepseek",
        include_str!("../../public/icons/models/deepseek.svg"),
    ),
    (
        "liquid",
        include_str!("../../public/icons/models/liquid.svg"),
    ),
    ("gemma", include_str!("../../public/icons/models/gemma.svg")),
    (
        "gemini",
        include_str!("../../public/icons/models/gemini.svg"),
    ),
    (
        "claude",
        include_str!("../../public/icons/models/claude.svg"),
    ),
    ("meta", include_str!("../../public/icons/models/meta.svg")),
    (
        "mistral",
        include_str!("../../public/icons/models/mistral.svg"),
    ),
    (
        "microsoft",
        include_str!("../../public/icons/models/microsoft.svg"),
    ),
    (
        "openai",
        include_str!("../../public/icons/models/openai.svg"),
    ),
];

fn starts_segment(text: &str, word: &str, then: impl Fn(Option<char>) -> bool) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        matches!(before, None | Some('/')) && then(after)
    })
}

pub fn family(model: &str) -> Option<&'static str> {
    let id = model.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|word| id.contains(word));
    let separated = |after: Option<char>| matches!(after, None | Some('-' | '/'));
    let lfm = id
        .match_indices("lfm")
        .any(|(at, _)| id[at + 3..].starts_with(|c: char| c.is_ascii_digit()));
    let liquid = id.match_indices("liquid").any(|(at, _)| {
        let rest = &id[at + 6..];
        let rest = rest.strip_prefix("ai").unwrap_or(rest);
        matches!(rest.chars().next(), None | Some('-' | '_' | '/'))
    });
    let phi = id.match_indices("phi").any(|(at, _)| {
        matches!(id[..at].chars().next_back(), None | Some('/' | '-'))
            && matches!(id[at + 3..].chars().next(), Some(c) if c == '-' || c.is_ascii_digit())
    });
    if has(&["qwen", "qwq"]) {
        Some("qwen")
    } else if has(&["deepseek"]) {
        Some("deepseek")
    } else if lfm || liquid {
        Some("liquid")
    } else if has(&["gemma"]) {
        Some("gemma")
    } else if has(&["gemini"]) {
        Some("gemini")
    } else if has(&["claude"]) {
        Some("claude")
    } else if has(&["llama"]) || starts_segment(&id, "meta", separated) {
        Some("meta")
    } else if has(&["mistral", "mixtral", "ministral", "codestral", "devstral"]) {
        Some("mistral")
    } else if phi || has(&["microsoft"]) {
        Some("microsoft")
    } else if has(&["gpt", "openai"])
        || ["o1", "o3", "o4"]
            .iter()
            .any(|word| starts_segment(&id, word, separated))
    {
        Some("openai")
    } else {
        None
    }
}

pub(super) async fn model_icon(Path(model): Path<String>) -> Response {
    let icon = family(&model).and_then(|name| {
        ICONS
            .iter()
            .find(|(family, _)| *family == name)
            .map(|(_, svg)| *svg)
    });
    match icon {
        Some(svg) => (
            [
                (header::CONTENT_TYPE, "image/svg+xml"),
                (header::CACHE_CONTROL, "private, max-age=3600"),
            ],
            svg,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::family;

    #[test]
    fn model_ids_map_to_their_family_and_unknown_models_have_none() {
        for (id, expected) in [
            ("qwen3.5-4b-q4", Some("qwen")),
            ("holo3.1-4b-q4", None),
            ("unsloth/Qwen3.8-27B-GGUF", Some("qwen")),
            ("lfm3-omni-2b", Some("liquid")),
            ("liquid-d1-3b", Some("liquid")),
            ("gemini-3.8-flash", Some("gemini")),
            ("gemma-4", Some("gemma")),
            ("meta/llama-4", Some("meta")),
            ("phi-4", Some("microsoft")),
            ("gpt-5", Some("openai")),
            ("o3-mini", Some("openai")),
            ("dots-studio/dots-3-note-preview:free", None),
        ] {
            assert_eq!(family(id), expected, "{id}");
        }
    }

    #[test]
    fn every_family_has_an_icon() {
        for (name, svg) in super::ICONS {
            assert!(svg.contains("<svg"), "{name}");
        }
    }
}
