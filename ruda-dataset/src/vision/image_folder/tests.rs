use super::decode::{parse_image_annotation, segmentation_mask_to_vec_usize};
use crate::Dataset;

use super::*;
const DATASET_ROOT: &str = "tests/data/image_folder";
const SEGMASK_ROOT: &str = "tests/data/segmask_folder";
const COCO_JSON: &str = "tests/data/dataset_coco.json";
const COCO_IMAGES: &str = "tests/data/image_folder_coco";

#[test]
pub fn image_folder_dataset() {
    let dataset = ImageFolderDataset::new_classification(DATASET_ROOT).unwrap();

    // Dataset has 3 elements
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(3), None);

    // Dataset elements should be: orange (0), red (1), red (1)
    assert_eq!(dataset.get(0).unwrap().annotation, Annotation::Label(0));
    assert_eq!(dataset.get(1).unwrap().annotation, Annotation::Label(1));
    assert_eq!(dataset.get(2).unwrap().annotation, Annotation::Label(1));
}

#[test]
pub fn image_folder_dataset_filtered() {
    let dataset = ImageFolderDataset::new_classification_with(DATASET_ROOT, &["jpg"]).unwrap();

    // Filtered dataset has 2 elements
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.get(2), None);

    // Dataset elements should be: orange (0), red (1)
    assert_eq!(dataset.get(0).unwrap().annotation, Annotation::Label(0));
    assert_eq!(dataset.get(1).unwrap().annotation, Annotation::Label(1));
}

#[test]
pub fn image_folder_dataset_with_items_sizes() {
    let root = Path::new(DATASET_ROOT);
    let items = vec![
        (root.join("orange").join("dot.jpg"), "orange".to_string()),
        (root.join("red").join("dot.jpg"), "red".to_string()),
        (root.join("red").join("dot.png"), "red".to_string()),
    ];
    let dataset =
        ImageFolderDataset::new_classification_with_items(items, &["orange", "red"]).unwrap();

    // Dataset has 3 elements
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(3), None);

    // Test item sizes

    assert_eq!(
        (
            dataset.get(0).unwrap().image_width,
            dataset.get(0).unwrap().image_height
        ),
        (1, 1)
    );
    assert_eq!(
        (
            dataset.get(1).unwrap().image_width,
            dataset.get(1).unwrap().image_height
        ),
        (1, 1)
    );
    assert_eq!(
        (
            dataset.get(2).unwrap().image_width,
            dataset.get(2).unwrap().image_height
        ),
        (1, 1)
    );
}

#[test]
pub fn image_folder_dataset_with_items() {
    let root = Path::new(DATASET_ROOT);
    let items = vec![
        (root.join("orange").join("dot.jpg"), "orange".to_string()),
        (root.join("red").join("dot.jpg"), "red".to_string()),
        (root.join("red").join("dot.png"), "red".to_string()),
    ];
    let dataset =
        ImageFolderDataset::new_classification_with_items(items, &["orange", "red"]).unwrap();

    // Dataset has 3 elements
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(3), None);

    // Dataset elements should be: orange (0), red (1), red (1)
    assert_eq!(dataset.get(0).unwrap().annotation, Annotation::Label(0));
    assert_eq!(dataset.get(1).unwrap().annotation, Annotation::Label(1));
    assert_eq!(dataset.get(2).unwrap().annotation, Annotation::Label(1));
}

#[test]
pub fn image_folder_dataset_multilabel() {
    let root = Path::new(DATASET_ROOT);
    let items = vec![
        (
            root.join("orange").join("dot.jpg"),
            vec!["dot".to_string(), "orange".to_string()],
        ),
        (
            root.join("red").join("dot.jpg"),
            vec!["dot".to_string(), "red".to_string()],
        ),
        (
            root.join("red").join("dot.png"),
            vec!["dot".to_string(), "red".to_string()],
        ),
    ];
    let dataset = ImageFolderDataset::new_multilabel_classification_with_items(
        items,
        &["dot", "orange", "red"],
    )
    .unwrap();

    // Dataset has 3 elements
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(3), None);

    // Dataset elements should be: [dot, orange] (0, 1), [dot, red] (0, 2), [dot, red] (0, 2)
    assert_eq!(
        dataset.get(0).unwrap().annotation,
        Annotation::MultiLabel(vec![0, 1])
    );
    assert_eq!(
        dataset.get(1).unwrap().annotation,
        Annotation::MultiLabel(vec![0, 2])
    );
    assert_eq!(
        dataset.get(2).unwrap().annotation,
        Annotation::MultiLabel(vec![0, 2])
    );
}

#[test]
#[should_panic]
pub fn image_folder_dataset_invalid_extension() {
    // Some invalid file extension
    let _ = ImageFolderDataset::new_classification_with(DATASET_ROOT, &["ico"]).unwrap();
}

#[test]
pub fn pixel_depth_try_into_u8() {
    let val = u8::MAX;
    let pix: u8 = PixelDepth::U8(val).try_into().unwrap();
    assert_eq!(pix, val);
}

#[test]
#[should_panic]
pub fn pixel_depth_try_into_u8_invalid() {
    let _: u8 = PixelDepth::U16(u8::MAX as u16 + 1).try_into().unwrap();
}

#[test]
pub fn pixel_depth_try_into_u16() {
    let val = u16::MAX;
    let pix: u16 = PixelDepth::U16(val).try_into().unwrap();
    assert_eq!(pix, val);
}

#[test]
#[should_panic]
pub fn pixel_depth_try_into_u16_invalid() {
    let _: u16 = PixelDepth::F32(u16::MAX as f32).try_into().unwrap();
}

#[test]
pub fn pixel_depth_try_into_f32() {
    let val = f32::MAX;
    let pix: f32 = PixelDepth::F32(val).try_into().unwrap();
    assert_eq!(pix, val);
}

#[test]
#[should_panic]
pub fn pixel_depth_try_into_f32_invalid() {
    let _: f32 = PixelDepth::U16(u16::MAX).try_into().unwrap();
}

#[test]
pub fn parse_image_annotation_label_string() {
    let classes = HashMap::from([("0".to_string(), 0_usize), ("1".to_string(), 1_usize)]);
    let anno = AnnotationRaw::Label("0".to_string());
    assert_eq!(
        parse_image_annotation(&anno, &classes),
        Annotation::Label(0)
    );
}

#[test]
pub fn parse_image_annotation_multilabel_string() {
    let classes = HashMap::from([
        ("0".to_string(), 0_usize),
        ("1".to_string(), 1_usize),
        ("2".to_string(), 2_usize),
    ]);
    let anno = AnnotationRaw::MultiLabel(vec!["0".to_string(), "2".to_string()]);
    assert_eq!(
        parse_image_annotation(&anno, &classes),
        Annotation::MultiLabel(vec![0, 2])
    );
}

#[test]
pub fn segmask_image_path_to_vec_usize() {
    let root = Path::new(SEGMASK_ROOT);

    // checkerboard mask
    const TEST_CHECKERBOARD_MASK_PATTERN: [u8; 64] = [
        1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1,
        2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1,
        2, 1, 2, 1,
    ];
    assert_eq!(
        TEST_CHECKERBOARD_MASK_PATTERN
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>(),
        segmentation_mask_to_vec_usize(&root.join("annotations").join("mask_checkerboard.png")),
    );

    // random 2 colors mask
    const TEST_RANDOM2COLORS_MASK_PATTERN: [u8; 64] = [
        1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1,
        1, 1, 1, 1,
    ];
    assert_eq!(
        TEST_RANDOM2COLORS_MASK_PATTERN
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>(),
        segmentation_mask_to_vec_usize(&root.join("annotations").join("mask_random_2colors.png")),
    );
    // random 3 colors mask
    const TEST_RANDOM3COLORS_MASK_PATTERN: [u8; 64] = [
        3, 1, 3, 3, 1, 1, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 3, 3, 3,
        2, 3, 2, 2, 3, 2, 3, 3, 1, 3, 1, 3, 3, 1, 1, 3, 2, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 1, 3, 3,
        2, 1, 2, 2,
    ];
    assert_eq!(
        TEST_RANDOM3COLORS_MASK_PATTERN
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>(),
        segmentation_mask_to_vec_usize(&root.join("annotations").join("mask_random_3colors.png")),
    );
}

#[test]
pub fn segmask_folder_dataset() {
    let root = Path::new(SEGMASK_ROOT);

    let items = vec![
        (
            root.join("images").join("image_checkerboard.png"),
            root.join("annotations").join("mask_checkerboard.png"),
        ),
        (
            root.join("images").join("image_random_2colors.png"),
            root.join("annotations").join("mask_random_2colors.png"),
        ),
        (
            root.join("images").join("image_random_3colors.png"),
            root.join("annotations").join("mask_random_3colors.png"),
        ),
    ];
    let dataset = ImageFolderDataset::new_segmentation_with_items(
        items,
        &[
            "foo", // 0
            "bar", // 1
            "baz", // 2
            "qux", // 3
        ],
    )
    .unwrap();

    // Dataset has 3 elements; each (image, annotation) is a single item
    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.get(3), None);

    // checkerboard mask
    const TEST_CHECKERBOARD_MASK_PATTERN: [u8; 64] = [
        1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1,
        2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1,
        2, 1, 2, 1,
    ];
    assert_eq!(
        dataset.get(0).unwrap().annotation,
        Annotation::SegmentationMask(SegmentationMask {
            mask: TEST_CHECKERBOARD_MASK_PATTERN
                .iter()
                .map(|&x| x as usize)
                .collect()
        })
    );
    // random 2 colors mask
    const TEST_RANDOM2COLORS_MASK_PATTERN: [u8; 64] = [
        1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1,
        1, 1, 1, 1,
    ];
    assert_eq!(
        dataset.get(1).unwrap().annotation,
        Annotation::SegmentationMask(SegmentationMask {
            mask: TEST_RANDOM2COLORS_MASK_PATTERN
                .iter()
                .map(|&x| x as usize)
                .collect()
        })
    );
    // random 3 colors mask
    const TEST_RANDOM3COLORS_MASK_PATTERN: [u8; 64] = [
        3, 1, 3, 3, 1, 1, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 3, 3, 3,
        2, 3, 2, 2, 3, 2, 3, 3, 1, 3, 1, 3, 3, 1, 1, 3, 2, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 1, 3, 3,
        2, 1, 2, 2,
    ];
    assert_eq!(
        dataset.get(2).unwrap().annotation,
        Annotation::SegmentationMask(SegmentationMask {
            mask: TEST_RANDOM3COLORS_MASK_PATTERN
                .iter()
                .map(|&x| x as usize)
                .collect()
        })
    );
}

#[test]
pub fn coco_detection_dataset() {
    let dataset = ImageFolderDataset::new_coco_detection(COCO_JSON, COCO_IMAGES).unwrap();
    assert_eq!(dataset.len(), 3); // we have only three images defined
    assert_eq!(dataset.get(3), None);

    const TWO_DOTS_AND_TRIANGLE_B1: BoundingBox = BoundingBox {
        coords: [3.125_172, 18.090_784, 10.960_11, 10.740_027],
        label: 0,
    };

    const TWO_DOTS_AND_TRIANGLE_B2: BoundingBox = BoundingBox {
        coords: [3.257_221_5, 3.037_139, 10.563_961, 10.828_06],
        label: 0,
    };

    const TWO_DOTS_AND_TRIANGLE_B3: BoundingBox = BoundingBox {
        coords: [15.097_662, 3.389_271, 12.632_737, 11.180_193],
        label: 1,
    };

    const DOTS_TRIANGLE_B1: BoundingBox = BoundingBox {
        coords: [3.125_172, 17.914_719, 10.828_06, 11.004_127],
        label: 0,
    };

    const DOTS_TRIANGLE_B2: BoundingBox = BoundingBox {
        coords: [15.273_727, 3.301_238, 12.192_573, 11.708_39],
        label: 1,
    };

    const ONE_DOT_B1: BoundingBox = BoundingBox {
        coords: [10.079_78, 9.595_598, 10.960_11, 11.356_258],
        label: 0,
    };

    for item in dataset.iter() {
        let file_name = Path::new(&item.image_path).file_name().unwrap();
        match item.annotation {
            // check if the number of bounding boxes is correct
            Annotation::BoundingBoxes(v) => {
                if file_name == "two_dots_and_triangle.jpg" {
                    assert_eq!(v.len(), 3);
                    assert!(v.contains(&TWO_DOTS_AND_TRIANGLE_B1));
                    assert!(v.contains(&TWO_DOTS_AND_TRIANGLE_B2));
                    assert!(v.contains(&TWO_DOTS_AND_TRIANGLE_B3));
                } else if file_name == "dot_triangle.jpg" {
                    assert_eq!(v.len(), 2);
                    assert!(v.contains(&DOTS_TRIANGLE_B1));
                    assert!(v.contains(&DOTS_TRIANGLE_B2));
                } else if file_name == "one_dot.jpg" {
                    assert_eq!(v.len(), 1);
                    assert!(v.contains(&ONE_DOT_B1));
                } else {
                    panic!("{}", format!("unexpected image name: {}", item.image_path));
                }
            }
            _ => panic!("unexpected annotation"),
        }
    }
}
