use super::processor::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};
use crate::GenerationError;
use std::path::Path;

fn decode_rgb(path: &Path) -> Result<image::RgbImage, String> {
    if image::ImageFormat::from_path(path).ok() != Some(image::ImageFormat::Jpeg) {
        return image::open(path).map(|image| image.into_rgb8()).map_err(|error| error.to_string());
    }
    use libjpeg_turbo_rs::{Decoder, DctMethod, PixelFormat};
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let mut decoder = Decoder::new(&bytes).map_err(|error| error.to_string())?;
    let header = decoder.header();
    let (width, height) = (header.width as u32, header.height as u32);
    let (format, color) = if header.components.len() == 1 {
        (PixelFormat::Grayscale, image::ColorType::L8)
    } else {
        (PixelFormat::Rgb, image::ColorType::Rgb8)
    };
    let mut limits = image::Limits::default();
    limits.check_dimensions(width, height).map_err(|error| error.to_string())?;
    limits.reserve_buffer(width, height, color).map_err(|error| error.to_string())?;
    decoder.set_output_format(format);
    decoder.set_dct_method(DctMethod::IsLow);
    let pixels = decoder.decode_image().map_err(|error| error.to_string())?.data;
    let image = match format {
        PixelFormat::Grayscale => image::GrayImage::from_raw(width,height,pixels).map(image::DynamicImage::ImageLuma8),
        _ => image::RgbImage::from_raw(width,height,pixels).map(image::DynamicImage::ImageRgb8),
    }.ok_or("JPEG decoded buffer does not match dimensions")?;
    Ok(image.into_rgb8())
}

impl Qwen35ImageProcessor {
    /// Decode JPEG/PNG files to RGB8 and run the same native resize/normalize/packing path.
    pub fn preprocess_files<P: AsRef<Path>>(
        &self, paths: &[P],
    ) -> Result<Qwen35PreparedImages, GenerationError> {
        let images = paths.iter().map(|path| {
            decode_rgb(path.as_ref())
                .map_err(|error| GenerationError(format!("cannot decode image {}: {error}",path.as_ref().display())))
        }).collect::<Result<Vec<_>,_>>()?;
        let views = images.iter().map(|image| Qwen35RgbImage {
            pixels: image.as_raw(), height:image.height() as usize, width:image.width() as usize,
        }).collect::<Vec<_>>();
        self.preprocess_rgb(&views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires RUDA_QWEN35_PROCESSOR_REFERENCE and RUDA_QWEN35_PNG_REFERENCE"]
    fn native_png_processor_independent_reference() {
        let reference = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_PROCESSOR_REFERENCE").unwrap());
        let pngs = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_PNG_REFERENCE").unwrap());
        let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(reference.join("processor.json")).unwrap()).unwrap();
        let png_meta: serde_json::Value = serde_json::from_slice(&std::fs::read(pngs.join("result.json")).unwrap()).unwrap();
        assert_eq!(png_meta["exact_rgb_roundtrip"], true);
        assert_eq!(png_meta["cases"].as_u64().unwrap() as usize, cases.as_array().unwrap().len());
        let mut first_two = Vec::new();
        for (i, case) in cases.as_array().unwrap().iter().enumerate() {
            let path = pngs.join(format!("image-{i}.png"));
            let image = image::open(&path).unwrap().into_rgb8();
            assert_eq!(image.width() as u64, case["width"].as_u64().unwrap());
            assert_eq!(image.height() as u64, case["height"].as_u64().unwrap());
            assert_eq!(*image.as_raw(), std::fs::read(reference.join(case["rgb"].as_str().unwrap())).unwrap());
            let processor = Qwen35ImageProcessor::from_json(&serde_json::to_vec(&case["config"]).unwrap()).unwrap();
            let actual = processor.preprocess_files(&[path]).unwrap();
            let expected: Vec<f32> = std::fs::read(reference.join(case["patches"].as_str().unwrap())).unwrap()
                .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            assert_eq!(actual.shape, serde_json::from_value::<[usize;2]>(case["shape"].clone()).unwrap());
            assert_eq!(actual.grids, serde_json::from_value::<Vec<[usize;3]>>(case["grids"].clone()).unwrap());
            assert_eq!(actual.patches, expected, "PNG case {i} must match the original independent preprocessing exactly");
            eprintln!("native PNG case={i}: RGB and {} patch values exact", actual.patches.len());
            if i < 2 { first_two.extend(actual.patches); }
        }
        let processor = Qwen35ImageProcessor::from_json(&serde_json::to_vec(&cases[0]["config"]).unwrap()).unwrap();
        assert_eq!(cases[0]["config"], cases[1]["config"]);
        let packed = processor.preprocess_files(&[pngs.join("image-0.png"),pngs.join("image-1.png")]).unwrap();
        assert_eq!(packed.patches, first_two);
        let grids: Vec<[usize;3]> = cases.as_array().unwrap()[..2].iter().flat_map(|case|
            serde_json::from_value::<Vec<[usize;3]>>(case["grids"].clone()).unwrap()).collect();
        assert_eq!(packed.grids, grids);
        assert_eq!(packed.shape, [grids.iter().map(|g| g.iter().product::<usize>()).sum(),1536]);
        for name in ["missing.png","invalid.png","truncated.png"] {
            assert!(processor.preprocess_files(&[pngs.join(name)]).is_err(), "{name}");
        }
        assert!(processor.preprocess_files::<&Path>(&[]).is_err());
    }

    #[test]
    #[ignore = "requires RUDA_QWEN35_PROCESSOR_REFERENCE"]
    fn native_jpeg_processor_independent_reference() {
        let directory = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_PROCESSOR_REFERENCE").unwrap());
        let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("processor.json")).unwrap()).unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ruda-dataset/tests/data/image_folder_coco");
        let processor = Qwen35ImageProcessor::from_json(&serde_json::to_vec(&cases[0]["config"]).unwrap()).unwrap();
        let images = [fixtures.join("one_dot.jpg"),fixtures.join("two_dots_and_triangle.jpg")];
        for (i,path) in images.iter().enumerate() {
            let decoded = decode_rgb(path).unwrap();
            let expected = std::fs::read(directory.join(cases[i]["rgb"].as_str().unwrap())).unwrap();
            assert_eq!(*decoded.as_raw(),expected,"JPEG case {i} decoded pixels must match exactly");
        }
        let packed = processor.preprocess_files(&images).unwrap();
        let expected = cases.as_array().unwrap()[..2].iter().flat_map(|case| {
            std::fs::read(directory.join(case["patches"].as_str().unwrap())).unwrap()
                .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        assert_eq!(packed.patches.len(),expected.len());
        let max = packed.patches.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        eprintln!("native JPEG decode+processor two-image max_abs={max}");
        assert_eq!(max,0.0,"native JPEG input must match the fixed independent image pixels and preprocessing");
        let mut offset = 0;
        for (i,path) in images.iter().enumerate() {
            let single = processor.preprocess_files(&[path]).unwrap();
            assert_eq!(single.grids[0],packed.grids[i]);
            assert_eq!(&packed.patches[offset..offset+single.patches.len()],single.patches);
            offset += single.patches.len();
        }
        assert_eq!(offset,packed.patches.len());
        assert!(processor.preprocess_files(&[fixtures.join("does-not-exist.jpg")]).is_err());
    }
}
