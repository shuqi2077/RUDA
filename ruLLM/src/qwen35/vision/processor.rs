use crate::{GenerationError, HuggingFaceLoadError};
use serde::Deserialize;
use std::path::Path;
use super::resize::bicubic_rgb;

/// Borrowed, interleaved RGB8 pixels; rows have no padding.
pub struct Qwen35RgbImage<'a> {
    pub pixels: &'a [u8],
    pub height: usize,
    pub width: usize,
}

/// CPU FP32 patch rows, ready for the existing vision model's F32 input tensor.
pub struct Qwen35PreparedImages {
    pub patches: Vec<f32>,
    pub shape: [usize; 2],
    pub grids: Vec<[usize; 3]>,
}

#[derive(Deserialize)]
struct Size { shortest_edge: usize, longest_edge: usize }

fn yes() -> bool { true }
fn rescale() -> f64 { 1.0 / 255.0 }
fn bicubic() -> u32 { 3 }

#[derive(Deserialize)]
struct Config {
    size: Size,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default = "yes")]
    do_resize: bool,
    #[serde(default = "yes")]
    do_rescale: bool,
    #[serde(default = "yes")]
    do_normalize: bool,
    #[serde(default = "rescale")]
    rescale_factor: f64,
    #[serde(default = "bicubic")]
    resample: u32,
}

pub struct Qwen35ImageProcessor { config: Config }

fn product(factors: &[usize]) -> Result<usize, GenerationError> {
    factors.iter().try_fold(1usize, |n, &v| n.checked_mul(v))
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| GenerationError("image dimensions exceed supported indexing".into()))
}

impl Qwen35ImageProcessor {
    pub(in crate::qwen35) fn validate_vision(&self, vision: &super::Qwen35VisionConfig) -> Result<(), GenerationError> {
        if vision.in_channels != 3 || self.config.patch_size != vision.patch_size
            || self.config.temporal_patch_size != vision.temporal_patch_size
            || self.config.merge_size != vision.spatial_merge_size
        {
            return Err(GenerationError("image processor does not match vision patch/merge configuration".into()));
        }
        Ok(())
    }

    pub fn from_huggingface(directory: impl AsRef<Path>) -> Result<Self, HuggingFaceLoadError> {
        let bytes = std::fs::read(directory.as_ref().join("preprocessor_config.json"))
            .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
        Self::from_json(&bytes)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, HuggingFaceLoadError> {
        let config: Config = serde_json::from_slice(bytes).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
        if config.size.shortest_edge == 0 || config.size.shortest_edge > config.size.longest_edge
            || config.size.longest_edge > i32::MAX as usize / 3
            || config.resample != 3
            || !config.rescale_factor.is_finite() || config.rescale_factor <= 0.0
            || config.image_mean.iter().any(|v| !v.is_finite())
            || config.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || product(&[config.patch_size, config.merge_size]).is_err()
            || product(&[3, config.temporal_patch_size, config.patch_size, config.patch_size]).is_err()
        {
            return Err(HuggingFaceLoadError("invalid or unsupported Qwen3.5 image processor configuration".into()));
        }
        Ok(Self { config })
    }

    pub fn resized_dimensions(&self, height: usize, width: usize) -> Result<[usize; 2], GenerationError> {
        product(&[height, width, 3])?;
        let c = &self.config;
        let factor = c.patch_size * c.merge_size;
        if c.do_resize && height.max(width) as f64 / height.min(width) as f64 > 200.0 {
            return Err(GenerationError("image aspect ratio exceeds 200".into()));
        }
        let (mut h, mut w) = (height, width);
        if c.do_resize {
            h = ((height as f64 / factor as f64).round_ties_even() as usize) * factor;
            w = ((width as f64 / factor as f64).round_ties_even() as usize) * factor;
            let area = h.checked_mul(w).ok_or_else(|| GenerationError("resized image area overflow".into()))?;
            if area > c.size.longest_edge {
                let beta = ((height * width) as f64 / c.size.longest_edge as f64).sqrt();
                h = factor.max((height as f64 / beta / factor as f64).floor() as usize * factor);
                w = factor.max((width as f64 / beta / factor as f64).floor() as usize * factor);
            } else if area < c.size.shortest_edge {
                let beta = (c.size.shortest_edge as f64 / (height * width) as f64).sqrt();
                h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
                w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
            }
        }
        product(&[h, w, 3])?;
        product(&[height, w, 3])?;
        if h % factor != 0 || w % factor != 0 {
            return Err(GenerationError("image dimensions must be patch/merge aligned".into()));
        }
        Ok([h, w])
    }

    /// Resize, normalize and pack still images in merge-block order with temporal repetition.
    pub fn preprocess_rgb(&self, images: &[Qwen35RgbImage<'_>]) -> Result<Qwen35PreparedImages, GenerationError> {
        if images.is_empty() { return Err(GenerationError("image batch must not be empty".into())); }
        let c = &self.config;
        let patch_width = product(&[3, c.temporal_patch_size, c.patch_size, c.patch_size])?;
        let mut grids = Vec::with_capacity(images.len());
        let mut total = 0usize;
        for image in images {
            if product(&[image.height, image.width, 3])? != image.pixels.len() {
                return Err(GenerationError("RGB buffer does not match image dimensions".into()));
            }
            let [h, w] = self.resized_dimensions(image.height, image.width)?;
            let grid = [1, h / c.patch_size, w / c.patch_size];
            total = total.checked_add(product(&grid)?).ok_or_else(|| GenerationError("image batch size overflow".into()))?;
            grids.push(grid);
        }
        let mut patches = Vec::with_capacity(product(&[total, patch_width])?);
        let reciprocal = (1.0 / c.rescale_factor) as f32;
        let mut means = c.image_mean;
        let mut stds = c.image_std;
        if c.do_rescale && c.do_normalize {
            for i in 0..3 { means[i] *= reciprocal; stds[i] *= reciprocal; }
        }
        for (image, &[_, gh, gw]) in images.iter().zip(&grids) {
            let (h, w) = (gh * c.patch_size, gw * c.patch_size);
            let pixels = bicubic_rgb(image.pixels, image.height, image.width, h, w);
            for by in 0..gh / c.merge_size {
                for bx in 0..gw / c.merge_size {
                    for iy in 0..c.merge_size {
                        for ix in 0..c.merge_size {
                            for channel in 0..3 {
                                for _ in 0..c.temporal_patch_size {
                                    for py in 0..c.patch_size {
                                        for px in 0..c.patch_size {
                                            let y = (by * c.merge_size + iy) * c.patch_size + py;
                                            let x = (bx * c.merge_size + ix) * c.patch_size + px;
                                            let mut value = pixels[(y * w + x) * 3 + channel] as f32;
                                            if c.do_normalize { value = (value - means[channel]) / stds[channel]; }
                                            else if c.do_rescale { value *= c.rescale_factor as f32; }
                                            patches.push(value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Qwen35PreparedImages { patches, shape: [total, patch_width], grids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> serde_json::Value {
        serde_json::json!({"size":{"shortest_edge":16,"longest_edge":256},
            "patch_size":2,"temporal_patch_size":2,"merge_size":2,
            "image_mean":[0.,0.,0.],"image_std":[1.,1.,1.],"do_rescale":false})
    }

    #[test]
    fn native_patch_order_and_temporal_repeat() {
        let processor = Qwen35ImageProcessor::from_json(&serde_json::to_vec(&config()).unwrap()).unwrap();
        let pixels = (0..4*8*3).map(|n| n as u8).collect::<Vec<_>>();
        let result = processor.preprocess_rgb(&[Qwen35RgbImage { pixels: &pixels, height:4, width:8 }]).unwrap();
        assert_eq!(result.grids, [[1,2,4]]);
        assert_eq!(result.shape, [8,24]);
        assert_eq!(&result.patches[..8], &[0.,3.,24.,27.,0.,3.,24.,27.]);
        assert_eq!(&result.patches[8..12], &[1.,4.,25.,28.]);
        assert_eq!(result.patches[24], 6.);
        assert_eq!(result.patches[48], 48.);
        assert_eq!(result.patches[96], 12.);
    }

    #[test]
    fn native_resize_rounding_and_invalid_inputs() {
        let processor = Qwen35ImageProcessor::from_json(&serde_json::to_vec(&config()).unwrap()).unwrap();
        assert_eq!(processor.resized_dimensions(10,6).unwrap(), [8,8]);
        assert_eq!(processor.resized_dimensions(2,2).unwrap(), [4,4]);
        assert_eq!(processor.resized_dimensions(32,32).unwrap(), [16,16]);
        assert!(processor.resized_dimensions(0,2).is_err());
        assert!(processor.resized_dimensions(1,201).is_err());
        assert!(processor.resized_dimensions(usize::MAX,2).is_err());
        assert!(processor.preprocess_rgb(&[]).is_err());
        assert!(processor.preprocess_rgb(&[Qwen35RgbImage{pixels:&[],height:4,width:4}]).is_err());
        let flat = vec![127; 5*7*3];
        let resized = bicubic_rgb(&flat,5,7,12,4);
        assert_eq!(resized, vec![127;12*4*3]);
        let mut invalid = config();
        invalid["merge_size"] = 0.into();
        assert!(Qwen35ImageProcessor::from_json(&serde_json::to_vec(&invalid).unwrap()).is_err());
    }

    #[test]
    #[ignore = "requires RUDA_QWEN35_PROCESSOR_REFERENCE"]
    fn native_processor_independent_reference() {
        let directory = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_PROCESSOR_REFERENCE").unwrap());
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("processor.json")).unwrap()).unwrap();
        for case in manifest.as_array().unwrap() {
            let processor = Qwen35ImageProcessor::from_json(&serde_json::to_vec(&case["config"]).unwrap()).unwrap();
            let bytes = std::fs::read(directory.join(case["rgb"].as_str().unwrap())).unwrap();
            let result = processor.preprocess_rgb(&[Qwen35RgbImage { pixels:&bytes,
                height:case["height"].as_u64().unwrap() as usize,width:case["width"].as_u64().unwrap() as usize }]).unwrap();
            let expected: Vec<f32> = std::fs::read(directory.join(case["patches"].as_str().unwrap())).unwrap()
                .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            assert_eq!(result.shape, serde_json::from_value::<[usize;2]>(case["shape"].clone()).unwrap());
            assert_eq!(result.grids, serde_json::from_value::<Vec<[usize;3]>>(case["grids"].clone()).unwrap());
            assert_eq!(result.patches.len(), expected.len());
            let max = result.patches.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
            eprintln!("native processor {}: max_abs={max}", case["rgb"]);
            assert_eq!(max, 0.0, "native RGB preprocessing must match independent CPU reference exactly");
        }
    }
}
