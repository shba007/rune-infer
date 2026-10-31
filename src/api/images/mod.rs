pub mod detection;
pub mod generation;
pub mod processing;

pub use detection::*;
pub use generation::*;
pub use processing::*;

// Match the existing image-processing edge bound. Model-specific latent/patch
// alignment remains the engine's responsibility.
pub(super) fn parse_generation_dimensions(size: &str) -> Option<(u32, u32)> {
    let parts: Vec<_> = size.split(['x', 'X', '×', '*']).map(str::trim).collect();
    if parts.len() != 2
        || parts
            .iter()
            .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    let width = parts[0].parse().ok()?;
    let height = parts[1].parse().ok()?;
    ((64..=4096).contains(&width) && (64..=4096).contains(&height)).then_some((width, height))
}

pub fn resolve_image_dimensions(
    size: Option<&str>,
    aspect_ratio: Option<&str>,
    default_res: Option<&str>,
) -> String {
    if let Some((width, height)) = size.and_then(parse_generation_dimensions) {
        return format!("{width}x{height}");
    }

    if let Some(ar) = aspect_ratio {
        let clean = ar.to_lowercase();
        if clean.contains("16:9") {
            return "1024x576".to_string();
        } else if clean.contains("9:16") {
            return "576x1024".to_string();
        } else if clean.contains("1:1") {
            return "1024x1024".to_string();
        } else if clean.contains("4:3") {
            return "1024x768".to_string();
        } else if clean.contains("3:4") {
            return "768x1024".to_string();
        } else if clean.contains("21:9") {
            return "1280x544".to_string();
        } else if clean.contains("9:21") {
            return "544x1280".to_string();
        }
    }

    if let Some(d) = default_res {
        let clean = d.split('(').next().unwrap_or(d).trim();
        if clean.contains('x') || clean.contains('×') || clean.contains('*') {
            return clean.replace(['×', '*'], "x");
        }
    }

    "1024x1024".to_string()
}

#[cfg(test)]
mod dimension_tests {
    use super::*;

    #[test]
    fn generation_dimensions_accept_hd_and_reject_unsafe_values() {
        for (value, expected) in [
            ("1080x1080", "1080x1080"),
            ("720x720", "720x720"),
            ("1280x720", "1280x720"),
            ("1920×1080", "1920x1080"),
            (" 1024 * 1024 ", "1024x1024"),
            ("256x256", "256x256"),
            ("4096x4096", "4096x4096"),
        ] {
            assert!(parse_generation_dimensions(value).is_some(), "{value}");
            assert_eq!(
                resolve_image_dimensions(Some(value), Some("16:9"), None),
                expected
            );
        }
        for value in [
            "",
            "auto",
            "0x1080",
            "-1x720",
            "32x32",
            "4097x720",
            "999999999999999x720",
            "720x720x720",
            "720.5x720",
            "+720x720",
            "x720",
        ] {
            assert!(parse_generation_dimensions(value).is_none(), "{value}");
        }
        assert_eq!(
            resolve_image_dimensions(None, Some("16:9"), None),
            "1024x576"
        );
    }
}
