use std::{collections::HashMap, path::PathBuf};

use image::ColorType;

use super::{
    Annotation, AnnotationRaw, ImageDatasetItem, ImageDatasetItemRaw, PathToImageDatasetItem,
    PixelDepth, SegmentationMask,
};
use crate::transform::Mapper;

pub(super) fn segmentation_mask_to_vec_usize(mask_path: &PathBuf) -> Vec<usize> {
    // Load image from disk
    let image = image::open(mask_path).unwrap();

    // Image as Vec<PixelDepth>
    // if rgb8 or rgb16, keep only the first channel assuming all channels are the same

    match image.color() {
        ColorType::L8 => image.into_luma8().iter().map(|&x| x as usize).collect(),
        ColorType::L16 => image.into_luma16().iter().map(|&x| x as usize).collect(),
        ColorType::Rgb8 => image
            .into_rgb8()
            .iter()
            .step_by(3)
            .map(|&x| x as usize)
            .collect(),
        ColorType::Rgb16 => image
            .into_rgb16()
            .iter()
            .step_by(3)
            .map(|&x| x as usize)
            .collect(),
        _ => panic!("Unrecognized image color type"),
    }
}

/// Parse the image annotation to the corresponding type.
pub(super) fn parse_image_annotation(
    annotation: &AnnotationRaw,
    classes: &HashMap<String, usize>,
) -> Annotation {
    // TODO: add support for other annotations
    // - [ ] Object bounding boxes
    // - [x] Segmentation mask
    // For now, only image classification labels and segmentation are supported.

    // Map class string to label id
    match annotation {
        AnnotationRaw::Label(name) => Annotation::Label(*classes.get(name).unwrap()),
        AnnotationRaw::MultiLabel(names) => Annotation::MultiLabel(
            names
                .iter()
                .map(|name| *classes.get(name).unwrap())
                .collect(),
        ),
        AnnotationRaw::SegmentationMask(mask_path) => {
            Annotation::SegmentationMask(SegmentationMask {
                mask: segmentation_mask_to_vec_usize(mask_path),
            })
        }
        AnnotationRaw::BoundingBoxes(v) => Annotation::BoundingBoxes(v.clone()),
    }
}

impl Mapper<ImageDatasetItemRaw, ImageDatasetItem> for PathToImageDatasetItem {
    /// Convert a raw image dataset item (path-like) to a 3D image array with a target label.
    fn map(&self, item: &ImageDatasetItemRaw) -> ImageDatasetItem {
        let annotation = parse_image_annotation(&item.annotation, &self.classes);

        // Load image from disk
        let image = image::open(&item.image_path).unwrap();

        // Save image dimensions for manipulation
        let img_width = image.width() as usize;
        let img_height = image.height() as usize;

        // Image as Vec<PixelDepth>
        let img_vec = match image.color() {
            ColorType::L8 => image
                .into_luma8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::La8 => image
                .into_luma_alpha8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::L16 => image
                .into_luma16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::La16 => image
                .into_luma_alpha16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgb8 => image
                .into_rgb8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::Rgba8 => image
                .into_rgba8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::Rgb16 => image
                .into_rgb16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgba16 => image
                .into_rgba16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgb32F => image
                .into_rgb32f()
                .iter()
                .map(|&x| PixelDepth::F32(x))
                .collect(),
            ColorType::Rgba32F => image
                .into_rgba32f()
                .iter()
                .map(|&x| PixelDepth::F32(x))
                .collect(),
            _ => panic!("Unrecognized image color type"),
        };

        ImageDatasetItem {
            image: img_vec,
            image_width: img_width,
            image_height: img_height,
            annotation,
            image_path: item.image_path.display().to_string(),
        }
    }
}
