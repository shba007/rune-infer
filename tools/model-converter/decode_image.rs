//! Decode a fixture with Infer's image crate before cross-runtime JPEG checks.
use std::{fs::OpenOptions, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: decode_image SOURCE_IMAGE NEW_PNG".into());
    }
    let mut reader = image::ImageReader::open(Path::new(&args[0]))?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    reader.limits(limits);
    let image = image::DynamicImage::ImageRgb8(reader.decode()?.to_rgb8());
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(Path::new(&args[1]))?;
    image.write_to(&mut output, image::ImageFormat::Png)?;
    Ok(())
}
