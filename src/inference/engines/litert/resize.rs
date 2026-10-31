//! RGB bicubic resize matching the reviewed Pillow 12 reference's rounding.
//! Horizontal then vertical, with 22-bit coefficients and byte clipping per pass.
//! Algorithm reference: https://github.com/python-pillow/Pillow/blob/12.0.0/src/libImaging/Resample.c
use image::RgbImage;

fn coefficients(input: u32, output: u32) -> Vec<(usize, Vec<i32>)> {
    let ratio = f64::from(input) / f64::from(output);
    let scale = ratio.max(1.0);
    (0..output)
        .map(|position| {
            let center = (f64::from(position) + 0.5) * ratio;
            let start = ((center - 2.0 * scale + 0.5) as i64).clamp(0, i64::from(input)) as usize;
            let end = ((center + 2.0 * scale + 0.5) as i64).clamp(0, i64::from(input)) as usize;
            let weights: Vec<f64> = (start..end)
                .map(|sample| {
                    let x = ((sample as f64 + 0.5 - center) / scale).abs();
                    if x < 1.0 {
                        ((1.5 * x - 2.5) * x) * x + 1.0
                    } else if x < 2.0 {
                        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            (
                start,
                weights
                    .iter()
                    .map(|weight| (weight / sum * f64::from(1 << 22)).round() as i32)
                    .collect(),
            )
        })
        .collect()
}

fn sample(weights: &[i32], mut pixel: impl FnMut(usize) -> u8) -> u8 {
    let sum = weights
        .iter()
        .enumerate()
        .fold(1_i64 << 21, |sum, (index, weight)| {
            sum + i64::from(pixel(index)) * i64::from(*weight)
        });
    (sum >> 22).clamp(0, 255) as u8
}

pub(super) fn reference_bicubic(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    let horizontal = coefficients(input.width(), width);
    let vertical = coefficients(input.height(), height);
    let intermediate = RgbImage::from_fn(width, input.height(), |x, y| {
        let (start, weights) = &horizontal[x as usize];
        image::Rgb(std::array::from_fn(|channel| {
            sample(weights, |index| {
                input.get_pixel((*start + index) as u32, y)[channel]
            })
        }))
    });
    RgbImage::from_fn(width, height, |x, y| {
        let (start, weights) = &vertical[y as usize];
        image::Rgb(std::array::from_fn(|channel| {
            sample(weights, |index| {
                intermediate.get_pixel(x, (*start + index) as u32)[channel]
            })
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn matches_pillow_12_rgb_goldens_for_downsample_upsample_and_identity() {
        let image = RgbImage::from_fn(13, 9, |x, y| {
            image::Rgb([
                ((x * 37 + y * 11) % 256) as u8,
                ((x * 7 + y * 53) % 256) as u8,
                if (x + y) % 2 == 0 { 0 } else { 255 },
            ])
        });
        for (width, height, expected) in [
            (
                7,
                5,
                "b3540827e600ac90d8f0a118c30aed6ebac5b959e46b39b5978087b5f3d19c20",
            ),
            (
                25,
                17,
                "2cbd8ade1251dbaeec8467534a4deaaa01cb6ecb9eaf84e568af5fcac6cd38fc",
            ),
            (
                13,
                9,
                "5ab0268f7b646f9ac12b0c4b2cfbd7c31c57adf890c0ee8a371b863cb4a85477",
            ),
        ] {
            assert_eq!(
                hex::encode(Sha256::digest(
                    reference_bicubic(&image, width, height).as_raw()
                )),
                expected
            );
        }
    }
}
