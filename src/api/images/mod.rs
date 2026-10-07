pub mod detection;
pub mod generation;
pub mod processing;

pub use detection::*;
pub use generation::*;
pub use processing::*;

pub fn resolve_image_dimensions(
    size: Option<&str>,
    aspect_ratio: Option<&str>,
    default_res: Option<&str>,
) -> String {
    if let Some(s) = size {
        if s.contains('x') || s.contains('×') || s.contains('*') {
            return s.to_string();
        }
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
