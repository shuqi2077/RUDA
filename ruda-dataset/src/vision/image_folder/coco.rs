use std::{collections::HashMap, fs, path::Path};

use serde_json::Value;

use super::{
    AnnotationRaw, BoundingBox, ImageDatasetItemRaw, ImageFolderDataset, ImageLoaderError,
    PathToImageDatasetItem,
};
use crate::InMemDataset;
use crate::transform::MapperDataset;

const BBOX_MIN_NUM_VALUES: usize = 4;

/// Retrieve all available classes from the COCO JSON
fn parse_coco_classes(
    json: &serde_json::Value,
) -> Result<HashMap<String, usize>, ImageLoaderError> {
    let mut classes = HashMap::new();

    if let Some(json_classes) = json["categories"].as_array() {
        for class in json_classes {
            let id = class["id"]
                .as_u64()
                .ok_or_else(|| ImageLoaderError::ParsingError("Invalid class ID".to_string()))
                .and_then(|v| {
                    usize::try_from(v).map_err(|_| {
                        ImageLoaderError::ParsingError("Class ID out of usize range".to_string())
                    })
                })?;

            let name = class["name"]
                .as_str()
                .filter(|&s| !s.is_empty())
                .ok_or_else(|| ImageLoaderError::ParsingError("Invalid class name".to_string()))?
                .to_string();

            classes.insert(name, id);
        }
    }

    if classes.is_empty() {
        return Err(ImageLoaderError::ParsingError(
            "No classes found in annotations".to_string(),
        ));
    }

    Ok(classes)
}

/// Retrieve annotations from COCO JSON
fn parse_coco_bbox_annotations(
    json: &serde_json::Value,
) -> Result<HashMap<u64, AnnotationRaw>, ImageLoaderError> {
    let mut annotations = HashMap::new();

    if let Some(json_annotations) = json["annotations"].as_array() {
        for annotation in json_annotations {
            let image_id = annotation["image_id"].as_u64().ok_or_else(|| {
                ImageLoaderError::ParsingError("Invalid image ID in annotation".into())
            })?;

            let class_id = annotation["category_id"]
                .as_u64()
                .ok_or_else(|| {
                    ImageLoaderError::ParsingError("Invalid class ID in annotations".to_string())
                })
                .and_then(|v| {
                    usize::try_from(v).map_err(|_| {
                        ImageLoaderError::ParsingError(
                            "Class ID in annotations out of usize range".to_string(),
                        )
                    })
                })?;

            let bbox_coords = annotation["bbox"]
                .as_array()
                .ok_or_else(|| ImageLoaderError::ParsingError("missing bbox array".to_string()))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| {
                            ImageLoaderError::ParsingError("invalid bbox value".to_string())
                        })
                        .map(|val| val as f32)
                })
                .collect::<Result<Vec<f32>, _>>()?;

            if bbox_coords.len() < BBOX_MIN_NUM_VALUES {
                return Err(ImageLoaderError::ParsingError(format!(
                    "not enough bounding box coordinates in annotation for image {image_id}",
                )));
            }

            let bbox = BoundingBox {
                coords: [
                    bbox_coords[0],
                    bbox_coords[1],
                    bbox_coords[2],
                    bbox_coords[3],
                ],
                label: class_id,
            };

            annotations
                .entry(image_id)
                .and_modify(|entry| {
                    if let AnnotationRaw::BoundingBoxes(bboxes) = entry {
                        bboxes.push(bbox.clone());
                    }
                })
                .or_insert_with(|| AnnotationRaw::BoundingBoxes(vec![bbox]));
        }
    }

    if annotations.is_empty() {
        return Err(ImageLoaderError::ParsingError(
            "no annotations found".to_string(),
        ));
    }

    Ok(annotations)
}

/// Retrieve all available images from the COCO JSON
fn parse_coco_images<P: AsRef<Path>>(
    images_path: &P,
    mut annotations: HashMap<u64, AnnotationRaw>,
    json: &serde_json::Value,
) -> Result<Vec<ImageDatasetItemRaw>, ImageLoaderError> {
    let mut images = Vec::new();
    if let Some(json_images) = json["images"].as_array() {
        for image in json_images {
            let image_id = image["id"].as_u64().ok_or_else(|| {
                ImageLoaderError::ParsingError("Invalid image ID in image list".to_string())
            })?;

            let file_name = image["file_name"]
                .as_str()
                .ok_or_else(|| ImageLoaderError::ParsingError("Invalid image ID".to_string()))?
                .to_string();

            let mut image_path = images_path.as_ref().to_path_buf();
            image_path.push(file_name);

            if !image_path.exists() {
                return Err(ImageLoaderError::IOError(format!(
                    "Image {} not found",
                    image_path.display()
                )));
            }

            let annotation = annotations
                .remove(&image_id)
                .unwrap_or_else(|| AnnotationRaw::BoundingBoxes(Vec::new()));

            images.push(ImageDatasetItemRaw {
                annotation,
                image_path,
            });
        }
    }

    if images.is_empty() {
        return Err(ImageLoaderError::ParsingError(
            "No images found in annotations".to_string(),
        ));
    }

    Ok(images)
}

impl ImageFolderDataset {
    /// Create a COCO detection dataset based on the annotations JSON and image directory.
    ///
    /// # Arguments
    ///
    /// * `annotations_json` - Path to the JSON file containing annotations in COCO format (for
    ///   example instances_train2017.json).
    ///
    /// * `images_path` - Path containing the images matching the annotations JSON.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_coco_detection<A: AsRef<Path>, I: AsRef<Path>>(
        annotations_json: A,
        images_path: I,
    ) -> Result<Self, ImageLoaderError> {
        let file = fs::File::open(annotations_json)
            .map_err(|e| ImageLoaderError::IOError(format!("Failed to open annotations: {e}")))?;
        let json: Value = serde_json::from_reader(file).map_err(|e| {
            ImageLoaderError::ParsingError(format!("Failed to parse annotations: {e}"))
        })?;

        let classes = parse_coco_classes(&json)?;
        let annotations = parse_coco_bbox_annotations(&json)?;
        let items = parse_coco_images(&images_path, annotations, &json)?;
        let dataset = InMemDataset::new(items);
        let mapper = PathToImageDatasetItem { classes };
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }
}
