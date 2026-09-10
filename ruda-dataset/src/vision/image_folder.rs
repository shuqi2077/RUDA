mod coco;
mod dataset;
mod decode;

#[cfg(test)]
mod tests;

use crate::InMemDataset;
use crate::transform::MapperDataset;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Image data type.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PixelDepth {
    /// 8-bit unsigned.
    U8(u8),
    /// 16-bit unsigned.
    U16(u16),
    /// 32-bit floating point.
    F32(f32),
}

impl TryFrom<PixelDepth> for u8 {
    type Error = &'static str;

    fn try_from(value: PixelDepth) -> Result<Self, Self::Error> {
        if let PixelDepth::U8(v) = value {
            Ok(v)
        } else {
            Err("Value is not u8")
        }
    }
}

impl TryFrom<PixelDepth> for u16 {
    type Error = &'static str;

    fn try_from(value: PixelDepth) -> Result<Self, Self::Error> {
        if let PixelDepth::U16(v) = value {
            Ok(v)
        } else {
            Err("Value is not u16")
        }
    }
}

impl TryFrom<PixelDepth> for f32 {
    type Error = &'static str;

    fn try_from(value: PixelDepth) -> Result<Self, Self::Error> {
        if let PixelDepth::F32(v) = value {
            Ok(v)
        } else {
            Err("Value is not f32")
        }
    }
}

/// Annotation type for different tasks.
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// Image-level label.
    Label(usize),
    /// Multiple image-level labels.
    MultiLabel(Vec<usize>),
    /// Object bounding boxes.
    BoundingBoxes(Vec<BoundingBox>),
    /// Segmentation mask.
    SegmentationMask(SegmentationMask),
}

/// Segmentation mask annotation.
/// For semantic segmentation, a mask has a single channel (C = 1).
/// For instance segmentation, there may be multiple masks per image (C >= 1).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationMask {
    /// Segmentation mask.
    pub mask: Vec<usize>,
}

/// Object detection bounding box annotation.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Coordinates in [x_min, y_min, width, height] format.
    pub coords: [f32; 4],

    /// Box class label.
    pub label: usize,
}

/// Image dataset item.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageDatasetItem {
    /// Image as a vector with a valid image type.
    pub image: Vec<PixelDepth>,

    /// Original source image width.
    pub image_width: usize,

    /// Original source image height.
    pub image_height: usize,

    /// Annotation for the image.
    pub annotation: Annotation,

    /// Original image source.
    pub image_path: String,
}

/// Raw annotation types.
#[derive(Deserialize, Serialize, Debug, Clone)]
enum AnnotationRaw {
    Label(String),
    MultiLabel(Vec<String>),
    BoundingBoxes(Vec<BoundingBox>),
    SegmentationMask(PathBuf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ImageDatasetItemRaw {
    /// Image path.
    image_path: PathBuf,

    /// Image annotation.
    annotation: AnnotationRaw,
}

impl ImageDatasetItemRaw {
    fn new<P: AsRef<Path>>(image_path: P, annotation: AnnotationRaw) -> ImageDatasetItemRaw {
        ImageDatasetItemRaw {
            image_path: image_path.as_ref().to_path_buf(),
            annotation,
        }
    }
}

struct PathToImageDatasetItem {
    classes: HashMap<String, usize>,
}

/// Error type for [ImageFolderDataset](ImageFolderDataset).
#[derive(Error, Debug)]
pub enum ImageLoaderError {
    /// Unknown error.
    #[error("unknown: `{0}`")]
    Unknown(String),

    /// I/O operation error.
    #[error("I/O error: `{0}`")]
    IOError(String),

    /// Invalid file error.
    #[error("Invalid file extension: `{0}`")]
    InvalidFileExtensionError(String),

    /// Parsing error.
    #[error("Parsing error: `{0}`")]
    ParsingError(String),
}

type ImageDatasetMapper =
    MapperDataset<InMemDataset<ImageDatasetItemRaw>, PathToImageDatasetItem, ImageDatasetItemRaw>;

/// A generic dataset to load images from disk.
pub struct ImageFolderDataset {
    dataset: ImageDatasetMapper,
}
