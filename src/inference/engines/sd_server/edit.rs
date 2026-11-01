use super::{SdServerEngine, native_sampling_args};
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ImageData, ImageGenerationResponse, Usage};
use base64::Engine;
use std::error::Error;

fn decode_image(bytes: &[u8]) -> Result<image::DynamicImage, &'static str> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| "Cannot read image")?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|_| "Image is invalid or exceeds 4096x4096 pixels")
}

fn png(image: &image::DynamicImage) -> Result<Vec<u8>, image::ImageError> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

pub(crate) fn validate_task(task: &InferenceTaskRequest) -> Result<(), &'static str> {
    let InferenceTaskRequest::ImageEdit {
        prompt,
        images,
        mask,
        size,
        response_format,
        n,
        steps,
        cfg_scale,
        seed,
    } = task
    else {
        return Ok(());
    };
    if prompt.trim().is_empty() {
        return Err("Image edits require a nonempty prompt");
    }
    if images.is_empty() || images.len() > 2 {
        return Err("Image edits currently support one or two verified source images");
    }
    if n.unwrap_or(1) != 1 {
        return Err("This edit cycle supports n=1 only");
    }
    if !matches!(response_format.as_deref(), None | Some("b64_json")) {
        return Err("Image edits currently support response_format=b64_json only");
    }
    if steps == &Some(0)
        || cfg_scale.is_some_and(|v| !v.is_finite() || v < 0.0)
        || seed.is_some_and(|v| v < -1)
    {
        return Err(
            "Edit steps must be positive, cfg_scale finite and nonnegative, and seed -1 or nonnegative",
        );
    }
    if prompt.contains("<sd_cpp_extra_args>")
        && (steps.is_some() || cfg_scale.is_some() || seed.is_some())
    {
        return Err("Specify sampling controls as fields or embedded native arguments, not both");
    }
    if let Some(size) = size {
        let dimensions = size
            .split_once('x')
            .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)));
        if !dimensions.is_some_and(|(w, h)| (64..=4096).contains(&w) && (64..=4096).contains(&h)) {
            return Err("Edit size must be WIDTHxHEIGHT with dimensions from 64 to 4096");
        }
    }
    let mut first_dimensions = None;
    for bytes in images {
        if !matches!(
            image::guess_format(bytes),
            Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP)
        ) {
            return Err("Source image must be PNG, JPEG or WEBP");
        }
        let image = decode_image(bytes)?;
        if first_dimensions.is_none()
            && mask.is_some()
            && image.color().bits_per_pixel() / u16::from(image.color().channel_count()) > 8
        {
            return Err(
                "Masked edits require an 8-bit first image to preserve protected pixels exactly",
            );
        }
        first_dimensions.get_or_insert((image.width(), image.height()));
        if size.is_none() && (image.width() < 64 || image.height() < 64) {
            return Err("Specify an edit size for a source image smaller than 64x64 pixels");
        }
    }
    if let Some(mask) = mask {
        if image::guess_format(mask).ok() != Some(image::ImageFormat::Png) {
            return Err(
                "Edit mask must be a PNG with an alpha channel; transparent pixels are editable",
            );
        }
        let mask = decode_image(mask)?;
        if !matches!(
            mask.color(),
            image::ColorType::La8 | image::ColorType::Rgba8
        ) {
            return Err(
                "Edit mask must have an 8-bit alpha channel; transparent pixels are editable",
            );
        }
        let dimensions = (mask.width(), mask.height());
        if Some(dimensions) != first_dimensions {
            return Err("Edit mask dimensions must match the first source image");
        }
        if size
            .as_ref()
            .is_some_and(|size| size != &format!("{}x{}", dimensions.0, dimensions.1))
        {
            return Err(
                "Masked edits must keep the first image's size to preserve protected pixels exactly",
            );
        }
        if dimensions.0 < 64
            || dimensions.1 < 64
            || dimensions.0 % 16 != 0
            || dimensions.1 % 16 != 0
        {
            return Err("Masked edit dimensions must be at least 64 and multiples of 16");
        }
        if prompt.contains("<sd_cpp_extra_args>") {
            return Err(
                "Masked edits require sampling controls as fields, without embedded native arguments",
            );
        }
    }
    Ok(())
}

impl SdServerEngine {
    pub(super) fn execute_edit(
        &self,
        task: &InferenceTaskRequest,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        execute_edit(
            &self.client,
            self.port,
            self.default_steps,
            self.default_cfg,
            task,
        )
    }
}

fn execute_edit(
    client: &reqwest::blocking::Client,
    port: u16,
    default_steps: u32,
    default_cfg: f32,
    task: &InferenceTaskRequest,
) -> Result<InferenceOutput, Box<dyn Error>> {
    validate_task(task)?;
    let InferenceTaskRequest::ImageEdit {
        prompt,
        images,
        mask,
        size,
        steps,
        cfg_scale,
        seed,
        ..
    } = task
    else {
        return Err("Expected image_edit task".into());
    };
    // Normalize uploads to the same decoded pixels used by mask preservation.
    // This also avoids depending on the subprocess's JPEG/WEBP decoder support.
    let references = images
        .iter()
        .map(|bytes| Ok(png(&decode_image(bytes)?)?))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let final_prompt = if prompt.contains("<sd_cpp_extra_args>") {
        prompt.clone()
    } else {
        let mut extra = serde_json::json!({"sample_params":native_sampling_args(steps.unwrap_or(default_steps),cfg_scale.unwrap_or(default_cfg),None)});
        if let Some(seed) = seed {
            extra["seed"] = (*seed).into();
        }
        // The pinned native edit route sets references but no init canvas. Both
        // are needed for masked denoising; native masks use white for editable.
        if mask.is_some() {
            extra["init_image"] = base64::engine::general_purpose::STANDARD
                .encode(&references[0])
                .into();
            // Qwen instruction edits need the full denoising schedule. The
            // source canvas is for masked preservation, not low-strength img2img.
            extra["strength"] = 1.0.into();
        }
        format!(
            "{}<sd_cpp_extra_args>{extra}</sd_cpp_extra_args>",
            prompt.trim()
        )
    };
    let mut form = reqwest::blocking::multipart::Form::new()
        .text("prompt", final_prompt)
        .text("n", "1");
    if let Some(size) = size {
        form = form.text("size", size.clone());
    }
    for bytes in references {
        form = form.part(
            "image[]",
            reqwest::blocking::multipart::Part::bytes(bytes)
                .file_name("reference.png")
                .mime_str("image/png")?,
        );
    }
    if let Some(mask) = mask {
        let alpha = decode_image(mask)?.to_rgba8();
        let native_mask = image::GrayImage::from_fn(alpha.width(), alpha.height(), |x, y| {
            image::Luma([if alpha.get_pixel(x, y)[3] == 0 {
                255
            } else {
                0
            }])
        });
        form = form.part(
            "mask",
            reqwest::blocking::multipart::Part::bytes(png(&image::DynamicImage::ImageLuma8(
                native_mask,
            ))?)
            .file_name("mask.png")
            .mime_str("image/png")?,
        );
    }
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/images/edits"))
        .multipart(form)
        .send()?;
    if !response.status().is_success() {
        return Err(format!("sd-server edit failed: {}", response.text()?).into());
    }
    let value: serde_json::Value = response.json()?;
    let items = value["data"]
        .as_array()
        .filter(|items| items.len() == 1)
        .ok_or("sd-server did not return one edited image")?;
    let data = items
        .iter()
        .map(|item| {
            let b64 = item["b64_json"]
                .as_str()
                .filter(|text| !text.is_empty())
                .ok_or("sd-server returned no edited image data")?;
            let b64 = if let Some(mask) = mask {
                preserve_protected_pixels(&images[0], mask, b64)?
            } else {
                b64.to_string()
            };
            Ok(ImageData {
                b64_json: Some(b64),
                url: None,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    Ok(InferenceOutput {
        response: InferenceTaskResponse::Image(ImageGenerationResponse {
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            data,
        }),
        usage: Usage::default(),
    })
}

fn preserve_protected_pixels(
    source: &[u8],
    mask: &[u8],
    generated: &str,
) -> Result<String, Box<dyn Error>> {
    let source = decode_image(source)?.to_rgba8();
    let mask = decode_image(mask)?.to_rgba8();
    let generated = base64::engine::general_purpose::STANDARD.decode(generated)?;
    let mut output = decode_image(&generated)?.to_rgba8();
    if output.dimensions() != source.dimensions() || mask.dimensions() != source.dimensions() {
        return Err("Native masked edit returned a different canvas size".into());
    }
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        if mask.get_pixel(x, y)[3] != 0 {
            *pixel = *source.get_pixel(x, y);
        }
    }
    Ok(base64::engine::general_purpose::STANDARD
        .encode(png(&image::DynamicImage::ImageRgba8(output))?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::Multipart, routing::post};
    use serde_json::json;

    fn edit() -> serde_json::Value {
        json!({"type":"image_edit", "prompt":"Make it blue", "images":[
            include_bytes!("../../../../tests/fixtures/studio/off-center.png").to_vec()
        ]})
    }

    fn mask() -> Vec<u8> {
        png(&image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(
            400,
            240,
            |x, _| image::Rgba([0, 0, 0, if x < 200 { 255 } else { 0 }]),
        )))
        .unwrap()
    }

    #[test]
    fn edit_options_reject_invalid_or_unimplemented_requests() {
        for (key, value, accepted) in [
            ("n", json!(1), true),
            ("n", json!(2), false),
            ("response_format", json!("url"), false),
            ("prompt", json!(" "), false),
            ("steps", json!(0), false),
            ("seed", json!(-1), true),
            ("seed", json!(-2), false),
            ("cfg_scale", json!(-0.1), false),
            ("size", json!("512x512"), true),
            ("size", json!("0x512"), false),
            ("images", json!([]), false),
            ("images", json!([[1, 2, 3]]), false),
        ] {
            let mut value_task = edit();
            value_task[key] = value;
            let task = serde_json::from_value(value_task).unwrap();
            assert_eq!(validate_task(&task).is_ok(), accepted, "{key}");
        }
        let mut value = edit();
        let image = value["images"][0].clone();
        for count in 1..=4 {
            value["images"] = json!(vec![image.clone(); count]);
            assert_eq!(
                validate_task(&serde_json::from_value(value.clone()).unwrap()).is_ok(),
                count <= 2
            );
        }
        let mut value = edit();
        value["mask"] = json!(mask());
        assert!(validate_task(&serde_json::from_value(value.clone()).unwrap()).is_ok());
        for (key, bad) in [
            ("mask", json!([1, 2, 3])),
            ("mask", edit()["images"][0].clone()), // RGB PNG has no alpha.
            ("size", json!("512x512")),
            (
                "prompt",
                json!("Edit<sd_cpp_extra_args>{}</sd_cpp_extra_args>"),
            ),
        ] {
            let mut bad_task = value.clone();
            bad_task[key] = bad;
            assert!(
                validate_task(&serde_json::from_value(bad_task).unwrap()).is_err(),
                "{key}"
            );
        }
    }

    #[tokio::test]
    async fn native_edit_transport_preserves_references_and_sampling_controls() {
        let (sent, received) = tokio::sync::oneshot::channel();
        let sent = std::sync::Arc::new(std::sync::Mutex::new(Some(sent)));
        let generated = png(&image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_pixel(400, 240, image::Rgba([0, 0, 255, 255])),
        ))
        .unwrap();
        let generated = base64::engine::general_purpose::STANDARD.encode(generated);
        let router = Router::new().route(
            "/v1/images/edits",
            post(move |mut form: Multipart| {
                let sent = sent.clone();
                let generated = generated.clone();
                async move {
                    let mut fields = Vec::new();
                    while let Some(field) = form.next_field().await.unwrap() {
                        let name = field.name().unwrap().to_string();
                        fields.push((name, field.bytes().await.unwrap().to_vec()));
                    }
                    sent.lock().unwrap().take().unwrap().send(fields).unwrap();
                    Json(json!({"data":[{"b64_json":generated}]}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let mut value = edit();
        let original = decode_image(include_bytes!(
            "../../../../tests/fixtures/studio/off-center.png"
        ))
        .unwrap();
        let encode = |format| {
            let mut bytes = std::io::Cursor::new(Vec::new());
            original.write_to(&mut bytes, format).unwrap();
            bytes.into_inner()
        };
        let jpeg = encode(image::ImageFormat::Jpeg);
        let source = decode_image(&jpeg).unwrap().to_rgba8();
        value["images"] = json!([jpeg, encode(image::ImageFormat::WebP)]);
        value["mask"] = json!(mask());
        value["size"] = json!("400x240");
        value["steps"] = json!(3);
        value["cfg_scale"] = json!(2.5);
        value["seed"] = json!(42);
        let task = serde_json::from_value(value).unwrap();
        let output = tokio::task::spawn_blocking(move || {
            execute_edit(&reqwest::blocking::Client::new(), port, 30, 5.0, &task)
                .map_err(|error| error.to_string())
        })
        .await
        .unwrap()
        .unwrap();
        let InferenceTaskResponse::Image(output) = output.response else {
            panic!("Wrong output")
        };
        let output = base64::engine::general_purpose::STANDARD
            .decode(output.data[0].b64_json.as_ref().unwrap())
            .unwrap();
        let output = decode_image(&output).unwrap().to_rgba8();
        for (x, y, pixel) in output.enumerate_pixels() {
            assert_eq!(
                pixel,
                if x < 200 {
                    source.get_pixel(x, y)
                } else {
                    &image::Rgba([0, 0, 255, 255])
                }
            );
        }
        let fields = received.await.unwrap();
        let images: Vec<_> = fields
            .iter()
            .filter(|(name, _)| name == "image[]")
            .collect();
        assert_eq!(images.len(), 2);
        assert_eq!(decode_image(&images[0].1).unwrap().to_rgba8(), source);
        assert_eq!(
            decode_image(&images[1].1).unwrap().to_rgba8(),
            original.to_rgba8()
        );
        let text = |key: &str| {
            String::from_utf8(
                fields
                    .iter()
                    .find(|(name, _)| name == key)
                    .unwrap()
                    .1
                    .clone(),
            )
            .unwrap()
        };
        assert_eq!(text("size"), "400x240");
        let native_mask = decode_image(&fields.iter().find(|(name, _)| name == "mask").unwrap().1)
            .unwrap()
            .to_luma8();
        assert_eq!(native_mask.get_pixel(0, 0)[0], 0);
        assert_eq!(native_mask.get_pixel(399, 0)[0], 255);
        let prompt = text("prompt");
        let extra: serde_json::Value = serde_json::from_str(
            prompt
                .split_once("<sd_cpp_extra_args>")
                .unwrap()
                .1
                .split_once("</sd_cpp_extra_args>")
                .unwrap()
                .0,
        )
        .unwrap();
        assert_eq!(
            extra,
            json!({"sample_params":{"sample_steps":3,"guidance":{"txt_cfg":2.5}},"seed":42,
                "init_image":base64::engine::general_purpose::STANDARD.encode(&images[0].1),"strength":1.0})
        );
        server.abort();
    }
}
